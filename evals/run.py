"""Compare registry plugins sequentially; reference labels never enter RAG inputs."""

import argparse
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys
from uuid import NAMESPACE_URL, uuid4, uuid5
from zoneinfo import ZoneInfo

from rag_core.config import AppError, Settings
from rag_core.corpus import digest
from rag_core.embeddings import CONTRACT
from rag_core.model import AnthropicChat
from rag_core.observability import get_telemetry, trace_metadata
from rag_core.registry import REGISTRY, UNAVAILABLE, get, load_all
from rag_core.store import connect, open_collection, read_documents
from .adapter import execute
from .dataset import DEFAULT, load
from .judges import AnthropicJudge, JudgeError
from .metrics import (
    GENERATION_KEYS,
    RETRIEVAL_KEYS,
    citation_metrics,
    generation_metrics,
    matches,
    retrieval_metrics,
)
from .report import aggregate, stages, write
from .schema import Passage


def validate_evidence(cases, documents):
    for case in cases:
        for expected in case.expected_evidence:
            if not any(matches(p, expected) for p in documents):
                raise AppError(
                    f"{case.id} has an evidence label absent from the prepared corpus: {expected.document} p.{expected.page} §{expected.clause}."
                )


def publish_scores(telemetry, output, scores, metadata):
    if telemetry is None:
        return None
    if not output.trace_id:
        return "Langfuse trace is missing; evaluation scores were not uploaded."
    try:
        for name, value in scores.items():
            if value is not None and name != "selected_count":
                telemetry.client.create_score(
                    score_id=uuid5(
                        NAMESPACE_URL, metadata["run_id"] + ":" + output.trace_id + ":" + name
                    ).hex,
                    trace_id=output.trace_id,
                    name="course." + name,
                    value=float(value),
                    data_type="NUMERIC",
                    metadata=metadata,
                )
        return telemetry.flush()
    except Exception as error:
        return f"Langfuse score upload failed ({type(error).__name__}); local results remain available."


def evaluate(
    settings,
    cases,
    architectures,
    collection,
    *,
    manifest,
    retrieval_only=False,
    judge=None,
    chat=None,
    telemetry_factory=None,
    progress=print,
):
    documents = [Passage.from_document(d) for d in read_documents(collection)]
    validate_evidence(cases, documents)
    run_id = uuid4().hex
    rows, checks = [], {}
    for name in architectures:
        current = replace(settings, architecture=name, temperature=0)
        architecture = get(name)
        check_error = None
        if architecture.check and not (retrieval_only and architecture.needs_llm_for_retrieval):
            try:
                checks[name] = architecture.check(current, collection)
            except Exception as error:
                check_error = (
                    str(error) if isinstance(error, AppError) else type(error).__name__
                )
        telemetry, trace_warning = None, None
        if current.langfuse:
            try:
                telemetry = (telemetry_factory or get_telemetry)(current)
            except Exception:
                trace_warning = (
                    "Langfuse initialization failed; no cloud scores for this architecture."
                )
        for case in cases:
            row = dict(
                architecture=name,
                case_id=case.id,
                type=case.type,
                language=case.language,
                split=case.split,
                outcome="completed",
                metrics={},
            )
            if retrieval_only and (architecture.needs_llm_for_retrieval or case.history):
                row.update(
                    outcome="skipped",
                    skip_reason="Model research or conversation rewriting is required; excluded from no-API retrieval evaluation.",
                )
                rows.append(row)
                continue
            if check_error:
                row.update(outcome="failed", error=check_error)
                rows.append(row)
                continue
            progress(f"{name} · {case.id}")
            try:
                output = execute(
                    case.request(),
                    current,
                    architecture,
                    collection,
                    chat,
                    retrieval_only=retrieval_only,
                    telemetry=telemetry,
                )
                scores = (
                    retrieval_metrics(case, output.passages)
                    if output.retrieval_complete
                    else dict.fromkeys((*RETRIEVAL_KEYS, "selected_count"))
                )
                if not retrieval_only:
                    scores.update(citation_metrics(case, output))
                row["output"] = output.model_dump()
                if output.error:
                    row.update(outcome="failed", error=output.error)
                judge_usage = None
                if not retrieval_only and output.status and judge:
                    try:
                        verdict, judge_usage = judge.evaluate(case, output, documents)
                        scores.update(
                            generation_metrics(
                                verdict, output, fact_count=len(case.expected_facts)
                            )
                        )
                        row["judgment"] = verdict.model_dump()
                    except Exception as error:
                        scores.update(dict.fromkeys(GENERATION_KEYS))
                        if isinstance(error, JudgeError):
                            row["rejected_judgment"] = error.verdict.model_dump()
                            row["judge_failure_reason"] = str(error)
                            judge_usage = error.usage
                        row.update(
                            outcome="failed",
                            judge_error=f"Judge failed ({type(error).__name__}); no generation quality score accepted.",
                        )
                row["metrics"] = scores
                row["stages"] = stages(output, judge_usage)
                row["upload_warning"] = (
                    trace_warning
                    or output.trace_warning
                    or publish_scores(
                        telemetry,
                        output,
                        scores,
                        dict(
                            run_id=run_id,
                            dataset_id=manifest["dataset_id"],
                            case_id=case.id,
                            split=case.split,
                            architecture=name,
                            judge=judge.configuration if judge else None,
                        ),
                    )
                )
            except Exception as error:
                row.update(
                    outcome="failed",
                    error=str(error) if isinstance(error, AppError) else type(error).__name__,
                )
            rows.append(row)
    configuration = {
        **trace_metadata(settings),
        "architectures": architectures,
        "split": cases[0].split,
        "mode": "retrieval-only" if retrieval_only else "full",
        "dataset_id": manifest["dataset_id"],
        "embedding_contract_id": digest(CONTRACT),
        "temperature": 0,
        "judge": judge.configuration if judge else None,
        "architecture_checks": checks,
        "cost": None,
        "cost_note": "No verified price table; usage is reported separately from monetary cost.",
        "cloud_upload_status": "queued; server readback not performed by this runner"
        if settings.langfuse
        else "disabled",
    }
    return dict(
        run_id=run_id,
        created_at=datetime.now(ZoneInfo("Europe/Athens")).isoformat(),
        configuration=configuration,
        complete=all(r["outcome"] == "completed" and not r.get("upload_warning") for r in rows),
        rows=rows,
        groups=aggregate(rows),
    )


def parser():
    defaults = Settings()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", default="all", help="all or one registered architecture")
    p.add_argument("--split", choices=["dev", "heldout"], default="dev")
    p.add_argument("--dataset", type=Path, default=DEFAULT)
    p.add_argument("--corpus", type=Path, default=Settings().corpus)
    p.add_argument("--collection", default=Settings().collection)
    for name in ("host", "port", "model", "top_k", "env_file"):
        value = getattr(defaults, name)
        p.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    p.add_argument("--option", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--retrieval-only", action="store_true")
    p.add_argument("--langfuse", action="store_true")
    p.add_argument(
        "--case",
        action="append",
        default=[],
        help="Explicit case ID; subset runs are smoke checks, not frozen baselines",
    )
    p.add_argument("--output", type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.output and (args.output.exists() or args.output.with_suffix(".md").exists()):
            raise AppError(
                "Report already exists; choose a new output path before running models."
            )
        extras = {}
        for option in args.option:
            key, sep, value = option.partition("=")
            if not sep or not key:
                raise AppError("Use --option KEY=VALUE.")
            try:
                extras[key] = json.loads(value)
            except ValueError:
                extras[key] = value
        args.top_k = extras.pop("top_k", args.top_k)
        settings = Settings(
            **{
                name: getattr(args, name)
                for name in (
                    "corpus",
                    "collection",
                    "host",
                    "port",
                    "model",
                    "top_k",
                    "env_file",
                    "langfuse",
                )
            },
            extras=extras,
            temperature=0,
        )
        # Review gate precedes service, model or credential access.
        cases, manifest = load(args.dataset, corpus=settings.corpus, split=args.split)
        if args.case:
            chosen = set(args.case)
            if chosen - {c.id for c in cases}:
                raise AppError("Requested case IDs are absent from the selected split.")
            cases = [c for c in cases if c.id in chosen]
        load_all()
        names = (
            sorted(REGISTRY, key=lambda name: REGISTRY[name].needs_llm_for_retrieval)
            if args.architecture == "all"
            else [args.architecture]
        )
        for name in names:
            get(name)
        collection = open_collection(connect(settings), settings)
        chat = None if args.retrieval_only else AnthropicChat(settings)
        judge = (
            None
            if args.retrieval_only
            else AnthropicJudge(AnthropicChat(settings, max_tokens=6000), settings.model)
        )
        report = evaluate(
            settings,
            cases,
            names,
            collection,
            manifest=manifest,
            retrieval_only=args.retrieval_only,
            chat=chat,
            judge=judge,
        )
        report["configuration"]["subset"] = bool(args.case)
        report["configuration"]["unavailable_plugins"] = dict(UNAVAILABLE)
        filename = (
            datetime.now(ZoneInfo("Europe/Athens")).strftime("%Y-%m-%d_%H%M%S")
            + "_"
            + report["run_id"][:8]
            + ".json"
        )
        path = args.output or Path("evals/reports") / filename
        write(report, path)
        print(f"Report: {path}\nTable: {path.with_suffix('.md')}")
        return int(
            any(r["outcome"] == "failed" or r.get("upload_warning") for r in report["rows"])
        )
    except (AppError, FileExistsError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
