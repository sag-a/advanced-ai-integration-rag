import json
from pathlib import Path

import pytest

from evals.dataset import DEFAULT, load, read_cases, refresh
from evals.run import main, validate_evidence
from rag_core.config import AppError
from rag_core.corpus import build_chunks, corpus_identity
from rag_core import prompts


def test_active_dataset_uses_only_the_original_pdf_and_reusable_course_cases():
    _, cases = read_cases(DEFAULT)
    loaded, manifest = load(corpus=Path("data/corpus"), split="dev")
    assert len(loaded) == len(cases) == 10
    assert {s["source"] for s in manifest["sources"]} == {
        "Magenta_Moments_Terms_Conditions_gr.pdf"
    }
    assert manifest["corpus_id"] == corpus_identity(Path("data/corpus"))
    assert {c.language for c in cases} == {"el", "en"}
    assert {c.type for c in cases} >= {"factual", "followup", "multihop", "abstain"}
    assert all(
        e.document == "Magenta_Moments_Terms_Conditions_gr.pdf"
        for c in cases
        for e in c.expected_evidence
    )
    source = prompts.REWRITE + prompts.ANSWER
    assert all(c.question not in source for c in cases)


def test_pending_heldout_gate_precedes_service_or_credentials(monkeypatch, tmp_path):
    rows, _ = read_cases(DEFAULT)
    dataset = tmp_path / "cases.jsonl"
    rows[0]["split"] = "heldout"
    dataset.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows))
    refresh(dataset, Path("data/corpus"))
    monkeypatch.setattr(
        "evals.run.connect", lambda *_: pytest.fail("No service access before review")
    )
    assert main(["--split", "heldout", "--retrieval-only", "--dataset", str(dataset)]) == 1
    with pytest.raises(AppError, match="review is pending"):
        load(dataset, corpus=Path("data/corpus"), split="heldout")


def test_changes_invalidate_approval_but_refresh_preserves_label_edits(tmp_path):
    rows, _ = read_cases(DEFAULT)
    file = tmp_path / "cases.jsonl"
    rows[0]["split"] = "heldout"
    file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    refresh(file, Path("data/corpus"))
    review = json.loads((tmp_path / "review.json").read_text())
    review.update(
        status="approved", reviewed_by="Test fixture reviewer", reviewed_on="2026-10-21"
    )
    (tmp_path / "review.json").write_text(json.dumps(review))
    assert len(load(file, corpus=Path("data/corpus"), split="heldout")[0]) == 1
    rows[0]["expected_facts"] = ["A corrected draft label"]
    file.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    with pytest.raises(AppError, match="changed"):
        load(file, corpus=Path("data/corpus"), split="heldout")
    refresh(file, Path("data/corpus"))
    assert read_cases(file)[1][0].expected_facts == ["A corrected draft label"]
    assert json.loads((tmp_path / "review.json").read_text())["status"] == "pending"


def test_all_evidence_resolves_to_original_physical_pages():
    # Uses the pinned local tokenizer; skip only when it has not been prepared.
    try:
        docs = build_chunks(Path("data/corpus"))
    except AppError as error:
        if "tokenizer is unavailable" in str(error):
            pytest.skip("Pinned local tokenizer has not been downloaded")
        raise
    from evals.schema import Passage

    _, cases = read_cases(DEFAULT)
    validate_evidence(cases, [Passage.from_document(d) for d in docs])
    assert len(docs) == 30
    assert max(d.metadata["token_count"] for d in docs) <= 512
