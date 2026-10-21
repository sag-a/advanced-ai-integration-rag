"""Jev, a decision model served by OpenRouter's decisions API (alpha).

Jev answers narrow, typed questions about a "state" with a calibrated probability
("noul", 0 = no, 1 = yes) instead of generating text. This architecture asks two:
does one passage help answer the question (reranking), and do the kept passages
contain the answer (the gate in front of the LLM). Requires OPENROUTER_API_KEY.
"""

from concurrent.futures import ThreadPoolExecutor

import httpx

from rag_core.config import AppError

URL = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.14"
PARALLEL = 8  # concurrent per-passage requests when Jev reranks

RELEVANT = {
    "relevant": {
        "type": "noul",
        "instructions": "Does the passage answer the question?",
        "criteria": {
            "true": "The passage contains the answer, or part of it",
            "false": "The passage does not help answer the question",
        },
    }
}
# "At least in part" matters: a strict "contains the answer" refused 3 of 30 answerable
# questions whose top-k held only some of the clauses they need (measured 2026-10-22).
GATE = {
    "answerable": {
        "type": "noul",
        "instructions": "Do these passages contain information that answers the question, fully or in part?",
        "criteria": {
            "true": "The passages answer the question, at least in part",
            "false": "Nothing in the passages answers the question",
        },
    }
}


def decide(settings, state, questions):
    """One decisions request; returns (answers, cost in USD)."""
    key = settings.env("OPENROUTER_API_KEY", required=True)
    try:
        response = httpx.post(
            URL,
            headers={"Authorization": f"Bearer {key}", "X-OpenRouter-Title": "rag-course"},
            json={"model": MODEL, "state": state, "questions": questions},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["answers"], (data.get("usage") or {}).get("cost") or 0.0
    except httpx.HTTPStatusError as error:
        raise AppError(
            f"Jev request failed with HTTP {error.response.status_code}. Check OPENROUTER_API_KEY and credits."
        ) from error
    except (httpx.HTTPError, KeyError, ValueError) as error:
        raise AppError(f"Jev request failed ({type(error).__name__}); no answer was accepted.") from error


def relevance(settings, question, passages):
    """P(passage helps answer the question) for each passage; returns (scores, cost)."""

    def one(passage):
        answers, cost = decide(settings, [f"Question: {question}", f"Passage: {passage}"], RELEVANT)
        return answers["relevant"]["noul"], cost

    with ThreadPoolExecutor(PARALLEL) as pool:
        results = list(pool.map(one, passages))
    return [score for score, _ in results], sum(cost for _, cost in results)


def route(settings, question, descriptions):
    """Pick one of `descriptions` ({name: what it covers}); returns (choice, probabilities, cost)."""
    questions = {
        "route": {
            "type": "choice",
            "instructions": "Which document collection can answer this question?",
            "criteria": descriptions,
        }
    }
    answers, cost = decide(settings, [f"Question: {question}"], questions)
    return answers["route"]["choice"], answers["route"]["probabilities"], cost


def gate(settings, question, passages):
    """P(the passages together contain the answer); returns (probability, cost)."""
    state = [f"Question: {question}"] + [
        f"Passage {number}: {passage}" for number, passage in enumerate(passages, 1)
    ]
    answers, cost = decide(settings, state, GATE)
    return answers["answerable"]["noul"], cost
