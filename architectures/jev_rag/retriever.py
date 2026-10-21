"""Dense search, a reranker, then a Jev gate that can abstain before the LLM runs.

0. Optional supervisor (`--option route=true`): Jev reads the question and picks
   the collection that covers its topic, or "neither", which abstains at once.
1. Vector search returns 20 candidates from that collection.
2. A reranker keeps the best top-k: the local bge cross-encoder (default) or Jev
   (`--option reranker=jev`, one OpenRouter call per candidate).
3. Jev estimates whether the kept passages answer the question. Below
   `gate_threshold` (default 0.5) the retriever returns no passages, so the graph
   answers "not in the documents" without calling the answer model.

The router only judges the topic; whether the answer is really in the documents
is left to the gate. A question about someone's own Hearts balance therefore
routes to Magenta Moments, and the gate abstains.
"""

from dataclasses import replace
from pathlib import Path
import time
from uuid import uuid4

from rich.console import Console

from rag_core import reranker
from rag_core.callbacks import report_stage
from rag_core.config import AppError
from rag_core.lesson_log import print_passages
from rag_core.retrieval import RetrievalResult
from rag_core.store import connect, open_collection, retrieve
from . import jev

CONSOLE = Console(stderr=True, highlight=False)
CANDIDATES = 20
RERANKERS = ("bge", "jev")
# name: (corpus folder, collection, what the router is told the collection covers)
ROUTES = {
    "magenta": (
        "data/corpus",
        "magenta_course_pdf_v1",
        "Magenta Moments loyalty programme terms and conditions (Greek): joining and leaving, "
        "member profile and personalised offers, Hearts points, coupons and codes, offers and "
        "partners, notifications, liability",
    ),
    "tariffs": (
        "data/corpus-tariffs",
        "cosmote_tariffs_v1",
        "COSMOTE residential calling and broadband price list from 2019: Double Play, Home "
        "Telephony and Fiberspeed plans, monthly prices, free minutes, international packages",
    ),
}
NEITHER = "A topic unrelated to both the loyalty programme and the calling and broadband plans"


def options(settings):
    name = settings.extras.get("reranker", "bge")
    threshold = settings.extras.get("gate_threshold", 0.5)
    routed = settings.extras.get("route", False)
    if name not in RERANKERS:
        raise AppError("Use --option reranker=bge or --option reranker=jev.")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise AppError("gate_threshold must be a number from 0 to 1.")
    if type(routed) is not bool:
        raise AppError("Use --option route=true or --option route=false.")
    return name, float(threshold), routed


def open_routes(settings):
    """Settings and an opened collection per route; fails early if one isn't prepared."""
    client = connect(settings)
    targets = {}
    for route, (corpus, collection, _) in ROUTES.items():
        target = replace(settings, corpus=Path(corpus), collection=collection)
        targets[route] = (target, open_collection(client, target))
    return targets


def make_retriever(settings, collection, log=None):
    name, threshold, routed = options(settings)
    targets = open_routes(settings) if routed else None

    def search(question):
        started = time.perf_counter()
        details = {"architecture": settings.architecture, "retrieval_id": uuid4().hex[:8]}
        target, target_collection, cost = settings, collection, 0.0
        if routed:
            descriptions = {route: about for route, (_, _, about) in ROUTES.items()} | {"neither": NEITHER}
            choice, probabilities, cost = jev.route(settings, question, descriptions)
            shown = ", ".join(f"{route} {p:.2f}" for route, p in probabilities.items())
            CONSOLE.print(f"Jev router: {shown} → {choice}")
            report_stage("jev-route", f"{choice} ({shown})")
            details.update(route=choice, route_probabilities=probabilities)
            if choice == "neither":
                CONSOLE.print("Jev router: no collection covers this topic → abstain before retrieval")
                details.update(gate_passed=False, jev_cost_usd=round(cost, 6), selected_ids=[])
                return RetrievalResult([], time.perf_counter() - started, details)
            target, target_collection = targets[choice]

        before = retrieve(target_collection, replace(target, top_k=CANDIDATES), question).documents
        texts = [d.page_content for d in before]
        if name == "jev":
            scores, rerank_cost = jev.relevance(settings, question, texts)
            cost += rerank_cost
            label = "Jev P(relevant)"
        else:
            scores = reranker.score(question, texts)
            label = "bge score"
        order = sorted(range(len(before)), key=lambda i: -scores[i])[: settings.top_k]
        kept = [before[i] for i in order]
        print_passages(
            CONSOLE,
            f"Before reranking: {len(before)} passages by vector distance (lower is closer)",
            before,
            "distance",
            [d.metadata["distance"] for d in before],
        )
        print_passages(
            CONSOLE,
            f"After {name} reranking: top {len(kept)} ({label}, higher is better)",
            kept,
            "score",
            [scores[i] for i in order],
            previous=[i + 1 for i in order],
        )
        probability, gate_cost = jev.gate(settings, question, [d.page_content for d in kept])
        passed = probability >= threshold
        decision = "answer with the LLM" if passed else "abstain: not in the documents"
        CONSOLE.print(
            f"Jev gate: P(these {len(kept)} passages answer the question, at least in part) = {probability:.2f}, "
            f"threshold {threshold:.2f} → {decision}"
        )
        report_stage("rerank", f"{name}: {len(before)} candidates → {len(kept)} kept")
        report_stage("jev-gate", f"P = {probability:.2f} → {decision}")
        details.update(
            collection=target.collection,
            reranker=name,
            rerank_scores=[round(scores[i], 4) for i in order],
            gate_probability=probability,
            gate_threshold=threshold,
            gate_passed=passed,
            jev_cost_usd=round(cost + gate_cost, 6),
            kept_ids=[d.id for d in kept],
            selected_ids=[d.id for d in kept] if passed else [],
        )
        return RetrievalResult(kept if passed else [], time.perf_counter() - started, details)

    return search


def check(settings, collection):
    name, threshold, routed = options(settings)
    settings.env("OPENROUTER_API_KEY", required=True)
    if name == "bge":
        reranker.load()
    result = {"reranker": name, "gate_threshold": threshold, "jev_model": jev.MODEL, "route": routed}
    if routed:
        result["routes"] = {route: target.collection for route, (target, _) in open_routes(settings).items()}
    return result
