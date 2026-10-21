"""Validate corpus identity and require a recorded human review for held-out runs."""

import argparse
from datetime import date
import json
from pathlib import Path

from pydantic import ValidationError

from rag_core.config import AppError, Settings
from rag_core.corpus import digest, source_manifest
from .schema import Case

DEFAULT = Path(__file__).parent / "datasets/cases.jsonl"


def read_cases(path):
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cases = [Case.model_validate(row) for row in rows]
    except (OSError, ValueError, ValidationError) as error:
        raise AppError(
            "Cannot read the typed evaluation dataset. Check the JSONL schema."
        ) from error
    if (
        not cases
        or len({c.id for c in cases}) != len(cases)
        or len({c.question for c in cases}) != len(cases)
    ):
        raise AppError("Dataset must contain unique case IDs and questions.")
    for case in cases:
        if case.expected_status == "answered" and (
            not case.expected_evidence or not case.expected_facts
        ):
            raise AppError(f"Answerable case {case.id} needs evidence and required facts.")
        if len({(e.document, e.page, e.clause) for e in case.expected_evidence}) != len(
            case.expected_evidence
        ):
            raise AppError(f"Duplicate evidence label in {case.id}.")
    return rows, cases


def load(path=DEFAULT, *, corpus, split):
    path = Path(path)
    rows, cases = read_cases(path)
    selected = [case for case in cases if case.split == split]
    if not selected:
        raise AppError(
            f"No cases in split {split}. This teaching dataset may contain development cases only."
        )
    try:
        manifest = json.loads(path.with_name("manifest.json").read_text())
        if manifest["dataset_id"] != digest(rows) or manifest["corpus_id"] != digest(
            source_manifest(corpus)
        ):
            raise AppError(
                "Dataset or corpus changed. Refresh the draft manifest and repeat review before evaluation."
            )
        if split == "heldout":
            review = json.loads(path.with_name("review.json").read_text())
            if (
                review.get("status") != "approved"
                or review.get("dataset_id") != manifest["dataset_id"]
                or not review.get("reviewed_by", "").strip()
            ):
                raise AppError(
                    "Held-out review is pending. Ask the instructor to review REVIEW.md and record approval in review.json. Dev runs are available now."
                )
            date.fromisoformat(review["reviewed_on"])
    except AppError:
        raise
    except (OSError, ValueError, KeyError) as error:
        raise AppError(
            "Dataset manifest or human review record is missing or invalid."
        ) from error
    return selected, manifest


def refresh(path, corpus):
    """Refresh metadata after editing labels; never approve a review automatically."""
    rows, cases = read_cases(path)
    sources = source_manifest(corpus)
    manifest = dict(
        version="draft-v1",
        case_count=len(cases),
        dataset_id=digest(rows),
        corpus_id=digest(sources),
        sources=sources,
        annotation_policy="Canonical PDF clauses; recall measures labeled source coverage, not every relevant paraphrase. Reused classroom scenarios are development examples, not an unseen generalization benchmark.",
    )
    path.with_name("manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    review_path = path.with_name("review.json")
    old = json.loads(review_path.read_text()) if review_path.exists() else {}
    if old.get("dataset_id") != manifest["dataset_id"]:
        old = dict(
            status="pending",
            dataset_id=manifest["dataset_id"],
            reviewed_by="",
            reviewed_on="",
            notes="Instructor review required before held-out execution.",
        )
        review_path.write_text(
            json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    lines = [
        "# Held-out review — draft v1",
        "",
        "Reviewer: course instructor. Review status is recorded in `review.json`.",
        "",
        "Check Greek/English wording, expected status, every material qualification and the canonical source map against physical PDF pages. Cases refer only to the supplied original Magenta Moments PDF. Flag ambiguous wording rather than guessing an intended answer.",
        "",
        f"Dataset identity: `{manifest['dataset_id']}`.",
        "",
        "After corrections to `cases.jsonl`, run `uv run --all-extras python -m evals.dataset refresh`. This regenerates this sheet and invalidates any approval for changed labels. After your human review, set `review.json` status to `approved`, with the matching dataset identity, your name in `reviewed_by` and ISO date in `reviewed_on`. Do not approve unreviewed cases.",
        "",
    ]
    if not any(c.split == "heldout" for c in cases):
        lines += [
            "No held-out cases are configured. The current cases are classroom development examples; no held-out baseline or approval is claimed.",
            "",
        ]
    for c in cases:
        if c.split != "heldout":
            continue
        lines += [f"## {c.id} · {c.language} · {c.expected_status}", "", c.question, ""]
        for h in c.history:
            lines += [f"> {h.role}: {h.text}", ""]
        lines += [
            "Canonical evidence: "
            + (
                "; ".join(
                    f"{e.document} · p.{e.page} · §{e.clause}" for e in c.expected_evidence
                )
                or "none required"
            ),
            "",
        ]
        lines += [f"- {fact}" for fact in c.expected_facts] or [
            "- No factual assertion required; score the refusal or clarification separately."
        ]
        lines += ["", "Review: ☐ accepted ☐ needs correction", ""]
    path.with_name("REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["refresh"])
    parser.add_argument("--dataset", type=Path, default=DEFAULT)
    parser.add_argument("--corpus", type=Path, default=Settings().corpus)
    args = parser.parse_args()
    result = refresh(args.dataset, args.corpus)
    print(
        "Refreshed draft metadata; review is never approved automatically. Dataset:",
        result["dataset_id"],
    )
