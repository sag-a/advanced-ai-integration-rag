from types import SimpleNamespace

import pandas as pd
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import evals.run as run
from rag_core.config import AppError, Settings
from rag_core.registry import load_all
from rag_core.retrieval import RetrievalResult
from tests.core.test_conversation import Chat, answer


def chunk(identifier, page, clause):
    return Document(
        id=identifier,
        page_content=f"Text of §{clause}",
        metadata={"source": "terms.pdf", "page_start": page, "page_end": page, "clause": clause},
    )


DOCUMENTS = [chunk("a", 1, "2.1"), chunk("b", 2, "3.3"), chunk("c", 2, "3.4")]


def case(identifier="c1", status="answered", history=(), evidence=((2, "3.3"),)):
    return {
        "id": identifier,
        "type": "factual",
        "question": "When do Hearts expire?",
        "history": list(history),
        "expected_status": status,
        "expected_evidence": [
            {"document": "terms.pdf", "page": page, "clause": clause} for page, clause in evidence
        ],
        "expected_facts": ["One year."] if status == "answered" else [],
    }


@pytest.fixture
def fake_app(monkeypatch):
    load_all()
    resources = SimpleNamespace(
        retrieve=lambda q: RetrievalResult(DOCUMENTS[1:], 0), chat=Chat(answer()), entry=None
    )
    monkeypatch.setattr(run, "create_resources", lambda settings: resources)
    monkeypatch.setattr(run, "connect", lambda settings: None)
    monkeypatch.setattr(run, "open_collection", lambda client, settings: None)
    monkeypatch.setattr(run, "read_documents", lambda collection: DOCUMENTS)
    return resources


def test_course_cases_load_and_unknown_ids_are_rejected():
    cases = run.load_cases()
    assert len(cases) == 21 and len({c["id"] for c in cases}) == 21
    assert [c["id"] for c in run.load_cases(ids=["s01-enrollment"])] == ["s01-enrollment"]
    with pytest.raises(AppError, match="Unknown case"):
        run.load_cases(ids=["nope"])


def test_evidence_labels_map_to_chunk_ids():
    assert run.evidence_ids(case(evidence=((1, "2.1"), (2, "3.3"))), DOCUMENTS) == ["a", "b"]
    with pytest.raises(AppError, match="no passage matches"):
        run.evidence_ids(case(evidence=((5, "9.9"),)), DOCUMENTS)


def test_answer_text_drops_reference_footers():
    message = AIMessage(content="Hearts last a year.\n[1] terms.pdf · PDF p. 2 · §3.3")
    assert run.answer_text(message) == "Hearts last a year."


def test_retrieval_only_scores_with_ragas_and_no_model(fake_app):
    rows = run.evaluate_architecture(
        "classic_rag",
        Settings(),
        [case(), case("c2", history=[{"role": "human", "text": "Hi"}])],
        retrieval_only=True,
        progress=lambda message: None,
    )
    assert rows[0]["id_based_context_recall"] == 1.0
    assert rows[0]["id_based_context_precision"] == 0.5
    assert rows[1]["note"].startswith("skipped")


def test_full_run_checks_status_and_scores_only_answerable_cases(fake_app, monkeypatch):
    seen = []

    def fake_score(samples, retrieval_only, settings):
        seen.extend(samples)
        return pd.DataFrame([{"faithfulness": 1.0} for _ in samples])

    monkeypatch.setattr(run, "score", fake_score)
    fake_app.chat = Chat(answer(), answer("Not in the documents.", [], "insufficient_evidence"))
    rows = run.evaluate_architecture(
        "classic_rag",
        Settings(),
        [case(), case("c2", status="insufficient_evidence", evidence=())],
        retrieval_only=False,
        progress=lambda message: None,
    )
    assert [r["status_match"] for r in rows] == [1.0, 1.0]
    assert rows[0]["faithfulness"] == 1.0 and "faithfulness" not in rows[1]
    assert len(seen) == 1 and seen[0].reference_context_ids == ["b"]
    assert seen[0].response == "Supported"


def test_failed_turn_is_reported_not_scored(fake_app, monkeypatch):
    monkeypatch.setattr(run, "score", lambda *args: pytest.fail("Nothing to score"))
    fake_app.chat = Chat(AppError("Provider unavailable"))
    rows = run.evaluate_architecture(
        "classic_rag", Settings(), [case()], retrieval_only=False, progress=lambda m: None
    )
    assert rows[0]["note"].startswith("failed")
