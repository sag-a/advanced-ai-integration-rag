"""One vector search, optionally reordered by a local cross-encoder reranker.

Turn the reranker on with `--option rerank=true`. It retrieves `rerank_candidates`
passages (default 20), scores each against the question and keeps the best top-k.
"""

from dataclasses import replace
import time
from uuid import uuid4

from rich.console import Console

from rag_core.callbacks import report_stage
from rag_core.config import AppError
from rag_core.lesson_log import print_passages
from rag_core import reranker
from rag_core.retrieval import RetrievalResult
from rag_core.store import retrieve

CONSOLE = Console(stderr=True, highlight=False)


def rerank_options(settings):
    enabled = settings.extras.get("rerank", False)
    candidates = settings.extras.get("rerank_candidates", 20)
    if type(enabled) is not bool:
        raise AppError("Use --option rerank=true or --option rerank=false.")
    if type(candidates) is not int or not settings.top_k <= candidates <= 20:
        raise AppError(f"rerank_candidates must be an integer from top-k ({settings.top_k}) to 20.")
    return enabled, candidates


def make_retriever(settings, collection, log=None):
    enabled, candidates = rerank_options(settings)

    def search(question):
        if not enabled:
            result = retrieve(collection, settings, question)
        else:
            started = time.perf_counter()
            first = retrieve(collection, replace(settings, top_k=candidates), question)
            before = first.documents
            scores = reranker.score(question, [d.page_content for d in before])
            order = sorted(range(len(before)), key=lambda i: -scores[i])[: settings.top_k]
            after = [
                before[i].model_copy(
                    update={"metadata": {**before[i].metadata, "rerank_score": scores[i]}}
                )
                for i in order
            ]
            print_passages(
                CONSOLE,
                f"Before reranking: {len(before)} passages by vector distance (lower is closer)",
                before,
                "distance",
                [d.metadata["distance"] for d in before],
            )
            print_passages(
                CONSOLE,
                f"After reranking: top {len(after)} by reranker score (higher is better)",
                after,
                "score",
                [scores[i] for i in order],
                previous=[i + 1 for i in order],
            )
            moved = sum(rank != i for rank, i in enumerate(order))
            report_stage("rerank", f"{len(before)} candidates → {len(after)} kept, {moved} moved")
            result = RetrievalResult(
                after,
                time.perf_counter() - started,
                {
                    "reranker": f"{reranker.MODEL}@{reranker.REVISION[:8]}",
                    "dense_ids": [d.id for d in before[: settings.top_k]],
                    "rerank_scores": [round(scores[i], 4) for i in order],
                },
            )
        result.details["architecture"] = settings.architecture
        result.details["retrieval_id"] = uuid4().hex[:8]
        result.details["selected_ids"] = [document.id for document in result.documents]
        return result

    return search


def check(settings, collection):
    enabled, candidates = rerank_options(settings)
    if enabled:
        reranker.load()
    return {"rerank": enabled, "rerank_candidates": candidates if enabled else None}
