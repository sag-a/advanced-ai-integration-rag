"""Local teaching hybrid: Chroma dense search + TF-IDF + RRF."""

from .keyword_search import Document, Hit, lexical_search, reciprocal_rank_fusion

MAX_DOCUMENTS = 1000


def retrieve_rankings(collection, query, *, method, top_k, candidate_k, where=None):
    """Return rankings and metadata over the same eligible collection records.

    Chroma's native hybrid Search API is Cloud-only:
    https://docs.trychroma.com/cloud/search-api/overview
    This small local example rebuilds TF-IDF from current stored text per query.
    """
    if method not in {"lexical", "hybrid", "compare"}:
        raise ValueError("Expected lexical, hybrid, or compare.")
    if not query.strip() or not 1 <= top_k <= candidate_k:
        raise ValueError("Provide a nonempty query and 1 <= top_k <= candidate_k.")
    records = collection.get(
        where=where, include=["documents", "metadatas"], limit=MAX_DOCUMENTS + 1
    )
    if len(records["ids"]) > MAX_DOCUMENTS:
        raise ValueError(
            f"Local keyword demo supports at most {MAX_DOCUMENTS} eligible records; "
            "use a persistent keyword index for a larger corpus."
        )
    documents = {}
    metadata = {}
    for identifier, text, tags in zip(
        records["ids"], records["documents"], records["metadatas"], strict=True
    ):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Keyword retrieval requires nonempty stored document text.")
        tags = tags or {}
        documents[identifier] = Document(id=identifier, text=text)
        metadata[identifier] = tags
    if not documents:
        return {"dense": [], "lexical": [], "hybrid": []}, metadata

    lexical = lexical_search(query, list(documents.values()), top_k=candidate_k)
    if method == "lexical":
        return {"lexical": lexical[:top_k]}, metadata

    # Chroma still embeds query_texts automatically through the configured Nomic function.
    results = collection.query(
        query_texts=[query], n_results=min(candidate_k, len(documents)), where=where,
        include=["distances"],
    )
    dense = []
    for identifier, distance in zip(results["ids"][0], results["distances"][0], strict=True):
        if identifier not in documents:
            raise ValueError("The collection changed during retrieval; rerun the query.")
        dense.append(Hit(documents[identifier], 1 - distance))
    hybrid = reciprocal_rank_fusion([lexical, dense], top_k=top_k)
    return {"dense": dense, "lexical": lexical, "hybrid": hybrid}, metadata


def search(collection, queries, *, method, top_k, candidate_k, where=None):
    print("\nLocal hybrid components: Chroma/Nomic + TF-IDF keywords + RRF (constant=60).")
    for query in queries:
        rankings, metadata = retrieve_rankings(
            collection, query, method=method, top_k=top_k, candidate_k=candidate_k, where=where
        )
        print(f"\nQuery: {query}\nMethod: {method} | Filter: {where}")
        methods = ("lexical", "dense", "hybrid") if method == "compare" else (method,)
        labels = {"lexical": "TF-IDF cosine", "dense": "Nomic cosine", "hybrid": "RRF score"}
        ranks = {
            name: {hit.document.id: rank for rank, hit in enumerate(hits, 1)}
            for name, hits in rankings.items()
        }
        for name in methods:
            print(f"\n=== {name.upper()} ===")
            if not rankings[name]:
                print("No keyword matches or no records matched the filter.")
            for rank, hit in enumerate(rankings[name][:top_k], 1):
                identifier = hit.document.id
                print(f"\n{rank}. {identifier} | {labels[name]}: {hit.score:.6f}")
                if name == "hybrid":
                    print(f"Component ranks: lexical={ranks['lexical'].get(identifier, 'absent')}, "
                          f"dense={ranks['dense'].get(identifier, 'absent')}")
                print(f"Metadata: {metadata[identifier]}\n{hit.document.text}")
    print("RRF combines rank positions, not raw similarity scores. Scores are not probabilities.")
    print("Hybrid is a comparison method; this demo does not establish a quality improvement.")
