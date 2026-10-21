from dataclasses import replace
from io import StringIO

import pytest
from langchain_core.documents import Document
from rich.console import Console

from architectures.classic_rag import retriever
from rag_core.config import AppError, Settings
from rag_core.retrieval import RetrievalResult


def chunk(identifier, distance):
    return Document(
        id=identifier,
        page_content=f"Section heading\n{identifier} Text of clause {identifier}.",
        metadata={"source": "t.pdf", "page_start": 1, "page_end": 1, "clause": identifier, "distance": distance},
    )


CANDIDATES = [chunk("1.1", 0.2), chunk("1.2", 0.3), chunk("1.3", 0.4), chunk("1.4", 0.5)]


@pytest.fixture
def setup(monkeypatch):
    calls, terminal = [], StringIO()

    def fake_retrieve(collection, settings, question):
        calls.append(settings.top_k)
        return RetrievalResult(CANDIDATES[: settings.top_k], 0.01)

    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(retriever, "CONSOLE", Console(file=terminal, width=200))
    # The cross-encoder prefers the passage the vector search ranked third.
    monkeypatch.setattr(retriever.reranker, "score", lambda question, texts: [0.1, -2.0, 5.0, -1.0][: len(texts)])
    return calls, terminal


def settings(**extras):
    return replace(Settings(), architecture="classic_rag", top_k=2, extras=extras)


def test_rerank_off_is_a_single_plain_vector_search(setup):
    calls, terminal = setup
    result = retriever.make_retriever(settings(), object())("Question")
    assert calls == [2]
    assert [d.id for d in result.documents] == ["1.1", "1.2"]
    assert terminal.getvalue() == ""


def test_rerank_widens_the_search_then_keeps_the_best_scored(setup):
    calls, terminal = setup
    result = retriever.make_retriever(settings(rerank=True, rerank_candidates=4), object())("Question")
    assert calls == [4]
    assert [d.id for d in result.documents] == ["1.3", "1.1"]
    assert result.documents[0].metadata["rerank_score"] == 5.0
    assert result.details["dense_ids"] == ["1.1", "1.2"]
    assert result.details["selected_ids"] == ["1.3", "1.1"]


def test_both_tables_are_printed_with_previous_ranks(setup):
    _, terminal = setup
    retriever.make_retriever(settings(rerank=True, rerank_candidates=4), object())("Question")
    printed = terminal.getvalue()
    assert "Before reranking: 4 passages" in printed and "After reranking: top 2" in printed
    after = printed.split("After reranking")[1]
    first_row = next(line for line in after.splitlines() if line.strip().startswith("1 "))
    assert first_row.split()[:2] == ["1", "3"]  # new rank 1 was rank 3
    assert "Section heading" not in printed  # the repeated heading is skipped in previews


@pytest.mark.parametrize(
    "extras, message",
    [
        ({"rerank": "yes"}, "rerank=true"),
        ({"rerank": True, "rerank_candidates": 1}, "rerank_candidates"),
        ({"rerank": True, "rerank_candidates": 21}, "rerank_candidates"),
    ],
)
def test_invalid_options_fail_before_any_search(setup, extras, message):
    with pytest.raises(AppError, match=message):
        retriever.make_retriever(settings(**extras), object())


def test_check_reports_missing_weights_with_the_download_command(monkeypatch):
    def missing(**kwargs):
        raise AppError(f"Reranker weights are not downloaded. Run: {retriever.reranker.DOWNLOAD}")

    monkeypatch.setattr(retriever.reranker, "load", missing)
    with pytest.raises(AppError, match="--download"):
        retriever.check(settings(rerank=True), object())
    assert retriever.check(settings(), object()) == {"rerank": False, "rerank_candidates": None}
