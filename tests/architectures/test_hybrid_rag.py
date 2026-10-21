from dataclasses import replace

from langchain_core.documents import Document

from architectures.hybrid_rag import retriever
from rag_core.config import Settings
from rag_core.registry import REGISTRY, load_all
from rag_core.retrieval import RetrievalResult


def chunk(identifier, text):
    return Document(
        id=identifier,
        page_content=text,
        metadata={"source": "t.pdf", "page_start": 1, "page_end": 1, "clause": identifier},
    )


DOCUMENTS = [
    chunk("a", "Double Play 24 XL with 420 free minutes to mobiles"),
    chunk("b", "Double Play 24 XL with 500 free minutes to mobiles"),
    chunk("c", "Home Telephony for pensioners"),
]


def test_reciprocal_rank_fusion_rewards_agreement_and_keeps_order_on_ties():
    assert retriever.fuse([["a", "b"], ["b", "c"]], top_k=3) == ["b", "a", "c"]
    assert retriever.fuse([["a"], ["c"]], top_k=1) == ["a"]


def test_keyword_match_lifts_a_passage_that_dense_search_ranked_low(monkeypatch):
    # Dense search ranks the 500-minute plan (b) third, behind an unrelated chunk (c).
    # Keywords rank b first, so fusion lifts it into the top 2 and drops c.
    monkeypatch.setattr(retriever, "read_documents", lambda collection: DOCUMENTS)
    monkeypatch.setattr(
        retriever, "retrieve", lambda *args: RetrievalResult([DOCUMENTS[0], DOCUMENTS[2], DOCUMENTS[1]], 0)
    )
    settings = replace(Settings(), architecture="hybrid_rag", top_k=2)
    result = retriever.make_retriever(settings, object())("Double Play 24 XL 500 minutes")
    assert [d.id for d in result.documents] == ["a", "b"]
    assert result.details["dense_ids"] == ["a", "c"]
    assert result.details["added_by_keywords"] == ["b"]


def test_tokens_keep_prices_plan_sizes_and_greek_without_accents():
    import re

    tokens = re.findall(retriever.TOKENS, "0,3072 €/min · 24 S · 13888 · Καρδιές.")
    assert {"0,3072", "24", "S", "13888", "Καρδιές"} <= set(tokens)


def test_hybrid_is_registered_without_model_calls():
    load_all()
    assert "hybrid_rag" in REGISTRY
    assert not REGISTRY["hybrid_rag"].needs_llm_for_retrieval
