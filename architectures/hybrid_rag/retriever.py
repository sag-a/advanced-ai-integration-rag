"""Dense Chroma search plus TF-IDF keywords, merged with reciprocal rank fusion."""

from dataclasses import replace
import time
from uuid import uuid4

from sklearn.feature_extraction.text import TfidfVectorizer

from rag_core.callbacks import report_stage
from rag_core.retrieval import RetrievalResult
from rag_core.store import read_documents, retrieve

CANDIDATES = 20  # per ranking, before fusion
RRF_CONSTANT = 60
# Keep prices ("0,3072"), plan sizes ("S", "XL") and numbers ("13888") as tokens.
TOKENS = r"(?u)\b\w(?:[\w,.]*\w)?\b"


def fuse(rankings, top_k):
    """Reciprocal rank fusion: each ranking adds 1 / (60 + rank) to an ID's score."""
    scores = {}
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, 1):
            scores[identifier] = scores.get(identifier, 0) + 1 / (RRF_CONSTANT + rank)
    return sorted(scores, key=lambda identifier: -scores[identifier])[:top_k]


def make_retriever(settings, collection, log=None):
    documents = read_documents(collection)
    by_id = {d.id: d for d in documents}
    # Accent folding matches Greek words written with and without tonos.
    vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode", token_pattern=TOKENS)
    matrix = vectorizer.fit_transform([d.page_content for d in documents])

    def search(question):
        started = time.perf_counter()
        candidates = min(CANDIDATES, len(documents))
        dense = retrieve(collection, replace(settings, top_k=candidates), question).documents
        scores = (matrix @ vectorizer.transform([question]).T).toarray().ravel()
        keywords = [documents[i].id for i in scores.argsort()[::-1][:candidates] if scores[i] > 0]
        dense_ids = [d.id for d in dense]
        selected = fuse([dense_ids, keywords], settings.top_k)
        dense_by_id = {d.id: d for d in dense}
        passages = [dense_by_id.get(i, by_id[i]) for i in selected]
        from_keywords = [i for i in selected if i not in dense_ids[: settings.top_k]]
        report_stage("hybrid-dense", f"{len(dense_ids)} candidates by meaning")
        report_stage("hybrid-keywords", f"{len(keywords)} candidates sharing words")
        report_stage(
            "hybrid-fusion", f"{len(passages)} passages, {len(from_keywords)} not in the dense top {settings.top_k}"
        )
        details = {
            "architecture": settings.architecture,
            "retrieval_id": uuid4().hex[:8],
            "selected_ids": selected,
            "dense_ids": dense_ids[: settings.top_k],
            "keyword_ids": keywords[: settings.top_k],
            "added_by_keywords": from_keywords,
        }
        return RetrievalResult(passages, time.perf_counter() - started, details)

    return search
