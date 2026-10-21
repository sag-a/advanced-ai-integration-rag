"""Keyword ranking and rank fusion for the local Chroma teaching example."""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class Hit:
    document: Document
    score: float


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _query_text(query: str) -> None:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")


def _unique_documents(documents: list[Document]) -> None:
    if len({doc.id for doc in documents}) != len(documents):
        raise ValueError("document IDs must be unique for embedding alignment")


def lexical_search(
    query: str, documents: list[Document], top_k: int = 3
) -> list[Hit]:
    """TF-IDF cosine baseline with Unicode tokens and exact equipment codes."""
    _query_text(query)
    _positive_int(top_k, "top_k")
    _unique_documents(documents)
    eligible = documents
    if not eligible:
        return []
    # Greek uppercase text commonly omits tonos; Unicode accent folding keeps
    # those forms comparable while preserving both Greek and Latin letters.
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", token_pattern=r"(?u)\b\w[\w-]*\b"
    )
    texts = [doc.text for doc in eligible]
    # Punctuation-only documents have no vocabulary and cannot match a query.
    analyzer = vectorizer.build_analyzer()
    if not any(analyzer(text) for text in texts):
        return []
    matrix = vectorizer.fit_transform(texts)
    scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
    hits = [
        Hit(doc, float(score))
        for doc, score in zip(eligible, scores, strict=True)
        if score > 0
    ]
    return sorted(hits, key=lambda hit: (-hit.score, hit.document.id))[:top_k]


def reciprocal_rank_fusion(
    rankings: list[list[Hit]], top_k: int = 3, k: int = 60
) -> list[Hit]:
    """Fuse ranks, not incomparable dense and lexical score scales."""
    _positive_int(top_k, "top_k")
    _positive_int(k, "k")
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for hit in ranking:
            identifier = hit.document.id
            if identifier in seen:
                continue
            seen.add(identifier)
            documents.setdefault(identifier, hit.document)
            scores[identifier] = scores.get(identifier, 0.0) + 1 / (k + len(seen))
    ordered = sorted(scores, key=lambda identifier: (-scores[identifier], identifier))
    return [
        Hit(documents[identifier], scores[identifier]) for identifier in ordered[:top_k]
    ]

