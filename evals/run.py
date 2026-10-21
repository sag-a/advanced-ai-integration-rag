"""Evaluate registered architectures on the course cases with Ragas.

Each case runs through the same graph as the CLI and UI. Ragas scores the
answerable cases; a status check covers refusals and clarifications.
"""

import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import re
import warnings

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")  # no usage analytics from Ragas
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ragas(\.|$)")
warnings.filterwarnings("ignore", message=r"Importing \w+ from 'ragas\.metrics' is deprecated")
warnings.filterwarnings("ignore", message=r"LangchainLLMWrapper is deprecated")

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

from app.runtime import create_resources  # noqa: E402
from rag_core.config import AppError, Settings  # noqa: E402
from rag_core.graph import build_graph, run_turn  # noqa: E402
from rag_core.model import AnthropicChat  # noqa: E402
from rag_core.registry import REGISTRY, get, load_all  # noqa: E402
from rag_core.store import connect, open_collection, read_documents  # noqa: E402

CASES = Path(__file__).parent / "datasets" / "cases.jsonl"
REPORTS = Path(__file__).parent / "reports"
FOOTER = re.compile(r"^\[\d+\] .+ · PDF p\. .*$", re.MULTILINE)


def load_cases(path=CASES, ids=()):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]
    if ids:
        unknown = set(ids) - {case["id"] for case in cases}
        if unknown:
            raise AppError("Unknown case IDs: " + ", ".join(sorted(unknown)))
        cases = [case for case in cases if case["id"] in ids]
    return cases


def evidence_ids(case, documents):
    """Chunk IDs whose source, page range and clause match the case's evidence labels."""
    ids = []
    for label in case["expected_evidence"]:
        found = [
            d.id
            for d in documents
            if d.metadata["source"] == label["document"]
            and d.metadata["page_start"] <= label["page"] <= d.metadata["page_end"]
            and d.metadata.get("clause") == label["clause"]
        ]
        if not found:
            raise AppError(f"{case['id']}: no passage matches p.{label['page']} §{label['clause']}.")
        ids += found
    return ids


def answer_text(message):
    """The answer without the reference footers the application appends."""
    return FOOTER.sub("", str(message.content)).strip()


def run_case(resources, settings, case, retrieval_only):
    if retrieval_only:
        return {"documents": resources.retrieve(case["question"]).documents}
    history = [
        (HumanMessage if turn["role"] == "human" else AIMessage)(content=turn["text"])
        for turn in case["history"]
    ]
    graph = build_graph(resources.chat, resources.retrieve, settings, entry=resources.entry)
    state = list(run_turn(graph, case["question"], history, settings))[-1]["data"]
    return {
        "documents": state["documents"],
        "response": answer_text(state["messages"][-1]),
        "status": state["answer_status"],
    }


def metrics_for(retrieval_only):
    from ragas.metrics import (
        FactualCorrectness,
        Faithfulness,
        IDBasedContextPrecision,
        IDBasedContextRecall,
        LLMContextRecall,
    )

    retrieval = [IDBasedContextRecall(), IDBasedContextPrecision()]
    if retrieval_only:
        return retrieval
    return retrieval + [Faithfulness(), LLMContextRecall(), FactualCorrectness()]


def score(samples, retrieval_only, settings):
    """Ragas scores, one row per sample. The judge is the course's Anthropic model."""
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper

    llm = None
    if not retrieval_only:
        llm = LangchainLLMWrapper(AnthropicChat(settings, max_tokens=4096).model)
    result = evaluate(
        EvaluationDataset(samples),
        metrics=metrics_for(retrieval_only),
        llm=llm,
        show_progress=False,
        raise_exceptions=False,
    )
    return result.to_pandas()


def evaluate_architecture(name, settings, cases, retrieval_only, progress=print):
    from ragas.dataset_schema import SingleTurnSample

    settings = replace(settings, architecture=name)
    architecture = get(name)
    resources = create_resources(settings)
    documents = read_documents(open_collection(connect(settings), settings))
    rows, samples = [], []
    for case in cases:
        row = {"architecture": name, "case": case["id"], "type": case["type"]}
        rows.append(row)
        if retrieval_only and (case["history"] or architecture.needs_llm_for_retrieval):
            row["note"] = "skipped: needs the model for history or research"
            continue
        progress(f"{name} · {case['id']}")
        try:
            output = run_case(resources, settings, case, retrieval_only)
        except AppError as error:
            row["note"] = f"failed: {error}"
            continue
        if not retrieval_only:
            row["status_match"] = float(output["status"] == case["expected_status"])
        if case["expected_status"] != "answered":
            continue  # nothing to score against; the status check covers it
        row["sample"] = len(samples)
        samples.append(
            SingleTurnSample(
                user_input=case["question"],
                response=output.get("response"),
                retrieved_contexts=[d.page_content for d in output["documents"]],
                retrieved_context_ids=[d.id for d in output["documents"]],
                reference="\n".join(case["expected_facts"]),
                reference_context_ids=evidence_ids(case, documents),
            )
        )
    if samples:
        scores = score(samples, retrieval_only, settings)
        names = tuple(m.name for m in metrics_for(retrieval_only))
        columns = [c for c in scores.columns if c.startswith(names)]
        for row in rows:
            if "sample" in row:
                row.update(scores.iloc[row.pop("sample")][columns].to_dict())
    return rows


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", default="all", help="all, or one registered architecture")
    p.add_argument(
        "--retrieval-only", action="store_true", help="ID-based retrieval scores, no model calls"
    )
    p.add_argument("--case", action="append", default=[], help="Run only this case ID (repeatable)")
    p.add_argument("--top-k", type=int, default=Settings().top_k)
    p.add_argument("--output", type=Path, help="CSV path (default: evals/reports/<time>.csv)")
    return p


def main(argv=None):
    import pandas as pd

    args = parser().parse_args(argv)
    try:
        cases = load_cases(ids=args.case)
        load_all()
        names = list(REGISTRY) if args.architecture == "all" else [get(args.architecture).name]
        settings = Settings(top_k=args.top_k, temperature=0)
        rows = [
            row
            for name in names
            for row in evaluate_architecture(name, settings, cases, args.retrieval_only)
        ]
    except AppError as error:
        print(error)
        return 1
    table = pd.DataFrame(rows)
    path = args.output or REPORTS / (datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print("\nMean per architecture (Ragas scores cover answerable cases only):")
    print(table.select_dtypes("number").groupby(table["architecture"]).mean().round(3).to_string())
    notes = table.get("note", pd.Series(dtype=str)).fillna("")
    if notes.any():
        print("\n" + table[notes != ""][["architecture", "case", "note"]].to_string(index=False))
    print(f"\nPer-case scores: {path}")
    return int(notes.str.startswith("failed").any())


if __name__ == "__main__":
    raise SystemExit(main())
