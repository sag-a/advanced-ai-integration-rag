"""Vector seeds, bounded breadth-first traversal, then original PDF passages."""

from dataclasses import replace
import math
import re
import time
import logging
from uuid import uuid4

from rag_core.config import AppError
from rag_core.retrieval import RetrievalResult
from rag_core.store import retrieve
from .index import load_index, normalized, read_documents
from .tracing import stage


def matches(alias, question):
    return bool(re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", question))


def expand(
    graph,
    documents,
    seeds,
    query,
    *,
    max_anchors=4,
    max_hops=2,
    max_edges=30,
    max_candidates=12,
):
    order = {d.id: i for i, d in enumerate(documents)}
    by_id = {d.id: d for d in documents}
    seed_ids = [d.id for d in seeds]
    linked = {
        key
        for key, entity in graph["entities"].items()
        if set(entity["chunk_ids"]) & set(seed_ids)
    }
    question = normalized(query)
    ambiguous = sorted(a for a in graph["ambiguous_aliases"] if matches(a, question))
    matched = {
        key
        for key in linked
        if any(
            normalized(a) not in graph["ambiguous_aliases"] and matches(normalized(a), question)
            for a in graph["entities"][key]["aliases"]
        )
    }
    anchors = sorted(
        linked,
        key=lambda key: (
            key not in matched,
            min(order[i] for i in graph["entities"][key]["chunk_ids"] if i in seed_ids),
            key,
        ),
    )[:max_anchors]
    frontier, visited = set(anchors), set(anchors)
    traversed, seen, candidates, graph_rank = [], set(), list(seed_ids), []
    for hop in range(1, max_hops + 1):
        edges = sorted(
            (
                r
                for r in graph["relationships"]
                if r["id"] not in seen and (r["source"] in frontier or r["target"] in frontier)
            ),
            key=lambda r: (order[r["chunk_id"]], r["id"]),
        )
        following = set()
        for edge in edges:
            if len(traversed) >= max_edges:
                break
            chunk = edge["chunk_id"]
            if chunk not in candidates and len(candidates) >= max_candidates:
                continue
            seen.add(edge["id"])
            traversed.append(
                {
                    **edge,
                    "hop": hop,
                    "page_start": by_id[chunk].metadata["page_start"],
                    "page_end": by_id[chunk].metadata["page_end"],
                    "clause": by_id[chunk].metadata.get("clause", ""),
                }
            )
            if chunk not in candidates:
                candidates.append(chunk)
            if chunk not in graph_rank:
                graph_rank.append(chunk)
            following.update({edge["source"], edge["target"]} - visited)
        visited.update(following)
        frontier = following
        if not frontier or len(traversed) >= max_edges:
            break
    added = [i for i in candidates if i not in seed_ids]
    return {
        "architecture": "graph_rag",
        "index_fingerprint": graph["fingerprint"],
        "seed_ids": seed_ids,
        "anchors": anchors,
        "ambiguous_query_aliases": ambiguous,
        "relationships": traversed,
        "entities": [
            graph["entities"][i]
            for i in sorted({e for r in traversed for e in (r["source"], r["target"])})
        ],
        "candidate_ids": candidates,
        "graph_order": graph_rank,
        "expansion_count": len(added),
        "traversed_count": len(traversed),
        "graph_used": bool(added),
        "explanation": f"Graph added {len(added)} candidate passages via {len(traversed)} relationships."
        if added
        else "Graph expansion contributed no additional passages; using vector candidates.",
        "limits": {
            "anchors": max_anchors,
            "hops": max_hops,
            "relationships": max_edges,
            "candidates": max_candidates,
        },
    }


def fuse(vector_order, graph_order, seeds, top_k):
    scores = {}
    for ranking in (vector_order, graph_order):
        for rank, identifier in enumerate(ranking, 1):
            scores[identifier] = scores.get(identifier, 0) + 1 / (60 + rank)
    ranked = sorted(
        scores,
        key=lambda i: (
            -scores[i],
            vector_order.index(i) if i in vector_order else len(vector_order),
            i,
        ),
    )
    first = seeds[:1]
    return (first + [i for i in ranked if i not in first])[:top_k]


class GraphRetriever:
    def __init__(self, collection, settings, *, reporter=None):
        self.collection, self.settings = collection, settings
        self.reporter = reporter
        self.documents = read_documents(collection)
        self.graph = load_index(self.documents, settings)
        self.by_id = {d.id: d for d in self.documents}

    def __call__(self, question):
        request_id = uuid4().hex[:8]
        reporter_available = self.reporter is not None

        def emit(event, **data):
            nonlocal reporter_available
            if reporter_available:
                try:
                    self.reporter(event, {"request_id": request_id, **data})
                except Exception:
                    reporter_available = False
                    logging.getLogger(__name__).warning(
                        "Classroom log unavailable; retrieval continues."
                    )

        emit("retrieval_started", top_k=self.settings.top_k)
        try:
            result = self._retrieve(question, emit)
        except Exception as error:
            emit("retrieval_failed", error_type=type(error).__name__)
            raise
        result.details["retrieval_id"] = request_id
        emit(
            "passages_selected",
            documents=result.documents,
            elapsed_seconds=result.elapsed_seconds,
            candidate_count=len(result.details["candidate_ids"]),
            hops_used=max((edge["hop"] for edge in result.details["relationships"]), default=0),
            **{
                key: result.details[key]
                for key in (
                    "seed_ids",
                    "graph_used",
                    "expansion_count",
                    "selected_expansion_count",
                    "traversed_count",
                    "index_fingerprint",
                    "limits",
                )
            },
        )
        return result

    def _retrieve(self, question, emit):
        started = time.perf_counter()
        warnings = []
        with stage(
            self.settings,
            "graph-vector-seeds",
            {"question": question, "top_k": min(3, self.settings.top_k)},
            warnings,
        ) as output:
            seed_result = retrieve(
                self.collection,
                replace(self.settings, top_k=min(3, self.settings.top_k)),
                question,
            )
            seeds, seconds = seed_result.documents, seed_result.elapsed_seconds
            output.update(passages=[d.model_dump() for d in seeds], elapsed_seconds=seconds)
        emit("seeds_selected", documents=seeds, elapsed_seconds=seconds)
        seed_time = time.perf_counter()
        if any(
            d.id not in self.by_id or d.page_content != self.by_id[d.id].page_content
            for d in seeds
        ):
            raise AppError(
                "Chroma passages changed after graph validation. Reopen the collection and use a matching index."
            )
        with stage(
            self.settings,
            "graph-traversal",
            {
                "seed_ids": [d.id for d in seeds],
                "index_fingerprint": self.graph["fingerprint"],
            },
            warnings,
        ) as output:
            details = expand(self.graph, self.documents, seeds, question)
            output.update(details)
        traversal_time = time.perf_counter()
        emit(
            "anchors_selected",
            anchors=[self.graph["entities"][key]["label"] for key in details["anchors"]],
            ambiguous_aliases=details["ambiguous_query_aliases"],
        )
        seen_chunks = set(details["seed_ids"])
        for hop in sorted({edge["hop"] for edge in details["relationships"]}):
            edges = [edge for edge in details["relationships"] if edge["hop"] == hop]
            added = {edge["chunk_id"] for edge in edges} - seen_chunks
            emit("hop_completed", hop=hop, relationships=edges, new_passages=len(added))
            seen_chunks.update(added)
        selection_started = time.perf_counter()
        with stage(
            self.settings,
            "graph-passage-selection",
            {"candidate_ids": details["candidate_ids"], "rrf_constant": 60},
            warnings,
        ) as output:
            if not details["graph_used"]:
                docs = seeds
                if self.settings.top_k > len(seeds):
                    docs = retrieve(self.collection, self.settings, question).documents
            else:
                candidates = details["candidate_ids"]
                try:
                    result = self.collection.query(
                        query_texts=[question],
                        n_results=len(candidates),
                        where={
                            "$and": [
                                {"corpus_id": {"$eq": self.graph["contract"]["corpus_id"]}},
                                {"chunk_id": {"$in": candidates}},
                            ]
                        },
                        include=["distances"],
                    )
                    ids, distances = result["ids"][0], result["distances"][0]
                    if (
                        set(ids) != set(candidates)
                        or len(ids) != len(candidates)
                        or not all(math.isfinite(d) for d in distances)
                    ):
                        raise ValueError("incomplete candidate ranking")
                    distance_map = dict(zip(ids, distances, strict=True))
                    selected = fuse(
                        ids,
                        details["graph_order"],
                        [d.id for d in seeds],
                        self.settings.top_k,
                    )
                    docs = [
                        self.by_id[i].model_copy(
                            update={
                                "metadata": {
                                    **self.by_id[i].metadata,
                                    "distance": distance_map[i],
                                }
                            }
                        )
                        for i in selected
                    ]
                except Exception as error:
                    raise AppError(
                        "Graph candidate ranking failed. Check Chroma; no answer was generated."
                    ) from error
            output.update(
                passages=[d.model_dump() for d in docs],
                elapsed_seconds=time.perf_counter() - selection_started,
            )
        details["trace_warnings"] = warnings
        details["selected_graph_ids"] = [
            d.id for d in docs if details["graph_used"] and d.id not in details["seed_ids"]
        ]
        details.update(
            selected_ids=[d.id for d in docs],
            seed_passages=[d.model_dump() for d in seeds],
            selected_passages=[d.model_dump() for d in docs],
            selected_expansion_count=sum(d.id not in details["seed_ids"] for d in docs)
            if details["graph_used"]
            else 0,
            seed_seconds=seconds,
            traversal_seconds=traversal_time - seed_time,
            selection_seconds=time.perf_counter() - selection_started,
        )
        return RetrievalResult(docs, time.perf_counter() - started, details)
