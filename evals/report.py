"""Per-type/language means include denominators; nulls never become perfect scores."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["architecture"], row["type"], row["language"])].append(row)
    result = []
    for (architecture, kind, language), items in sorted(groups.items()):
        keys = sorted({key for item in items for key in item.get("metrics", {})})
        scores = {}
        for key in keys:
            values = [
                item["metrics"][key]
                for item in items
                if item.get("metrics", {}).get(key) is not None
            ]
            scores[key] = dict(mean=mean(values) if values else None, n=len(values))
        elapsed = [r["output"]["elapsed_seconds"] for r in items if r.get("output")]
        result.append(
            dict(
                architecture=architecture,
                type=kind,
                language=language,
                attempted=len(items),
                completed=sum(r["outcome"] == "completed" for r in items),
                failed=sum(r["outcome"] == "failed" for r in items),
                skipped=sum(r["outcome"] == "skipped" for r in items),
                metrics=scores,
                mean_seconds=mean(elapsed) if elapsed else None,
            )
        )
    return result


def stages(output, judge_usage=None):
    obs = output.observation
    details = obs.get("retrieval_details", {})

    def stage(value):
        if not value:
            return None
        return dict(
            completed_calls=value.get("completed_calls", 1),
            method=value.get("method", "model"),
            attempts=value.get("attempts", []),
            elapsed_seconds=value.get("elapsed_seconds"),
            usage=value.get("usage", {}),
        )

    research = details.get("research_usage", [])
    return dict(
        rewrite=stage(obs.get("rewrite")),
        research=dict(
            completed_calls=details.get("model_calls"),
            elapsed_seconds=obs.get("retrieval_seconds"),
            usage_records=research,
        )
        if "model_calls" in details
        else None,
        generation=stage(obs.get("generation")),
        judge=stage(judge_usage),
        note="Logical calls and returned usage; provider retry attempts or failed-call token usage may be unavailable. Indexing usage is separate. No prices assumed.",
    )


def markdown(report):
    config = report["configuration"]
    lines = [
        "# RAG evaluation report",
        "",
        f"Run: `{report['run_id']}` · {report['created_at']} · split **{config['split']}** · mode **{config['mode']}**.",
        "",
        f"Dataset `{config['dataset_id']}` · corpus `{config['corpus_id']}` · final top-k {config['top_k']}.",
        "",
        f"Completed without execution/judge errors: **{report['complete']}**. A successful run does not establish correctness or an architecture improvement.",
        "",
        "Scores are means with scored-case counts. Null is not applicable or unavailable; failures and skips remain visible. Retrieval measures canonical labeled source coverage, not all semantically relevant paraphrases. Factual correctness is F1 of judged claim precision and required-fact recall.",
        "",
        "| Architecture | Type | Lang | Done / attempted | Recall@k | Precision@k | Factual F1 | Fact recall | Faithfulness | Citation support | Status match | Mean seconds |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:

        def score(key):
            m = group["metrics"].get(key, {})
            return f"{m['mean']:.3f} (n={m['n']})" if m.get("mean") is not None else "N/A"

        seconds = f"{group['mean_seconds']:.2f}" if group["mean_seconds"] is not None else "N/A"
        lines.append(
            "| "
            + " | ".join(
                [
                    group["architecture"],
                    group["type"],
                    group["language"],
                    f"{group['completed']} / {group['attempted']}",
                    score("recall_at_k"),
                    score("precision_at_k"),
                    score("factual_correctness"),
                    score("factual_recall"),
                    score("faithfulness"),
                    score("citation_support"),
                    score("status_match"),
                    seconds,
                ]
            )
            + " |"
        )
    problems = [
        r
        for r in report["rows"]
        if r.get("error")
        or r.get("judge_error")
        or r.get("skip_reason")
        or r.get("upload_warning")
    ]
    lines += ["", "## Failures, skips and upload warnings", ""]
    lines += [
        f"- {r['architecture']} / {r['case_id']}: {r.get('error') or r.get('judge_error') or r.get('skip_reason') or r.get('upload_warning')}"
        for r in problems
    ] or ["None recorded."]
    lines += [
        "",
        "Raw answers, source passages, labels, stage timing and available token usage are in the paired JSON. Monetary cost is unreported because no verified price table was supplied. Human review is still needed for judge disagreements and language/conversation quality.",
        "",
    ]
    return "\n".join(lines)


def write(report, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(".md").exists():
        raise FileExistsError("Report already exists; choose a new output path.")
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    path.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
