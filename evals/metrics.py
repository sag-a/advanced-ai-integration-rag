"""Compute scores from evidence matches and atomic labels, never LLM numbers."""

from .schema import Case, Evidence, Judgment, Passage, RunOutput

RETRIEVAL_KEYS = ("recall_at_k", "precision_at_k", "hit_at_k", "reciprocal_rank")
GENERATION_KEYS = (
    "factual_precision",
    "factual_recall",
    "factual_correctness",
    "faithfulness",
    "citation_support",
)


def matches(passage: Passage, evidence: Evidence):
    return (
        passage.source == evidence.document
        and passage.clause == evidence.clause
        and passage.page_start <= evidence.page <= passage.page_end
    )


def retrieval_metrics(case: Case, passages: list[Passage]):
    # First occurrence wins; repeated IDs cannot increase credit.
    by_id = {}
    for passage in passages:
        by_id.setdefault(passage.id, passage)
    unique = list(by_id.values())
    result = {"selected_count": len(unique)}
    if not case.expected_evidence:
        return {**result, **dict.fromkeys(RETRIEVAL_KEYS)}
    relevant = [any(matches(p, e) for e in case.expected_evidence) for p in unique]
    first = next((rank for rank, hit in enumerate(relevant, 1) if hit), None)
    return {
        **result,
        "recall_at_k": sum(any(matches(p, e) for p in unique) for e in case.expected_evidence)
        / len(case.expected_evidence),
        "precision_at_k": sum(relevant) / len(unique) if unique else 0.0,
        "hit_at_k": float(first is not None),
        "reciprocal_rank": 1 / first if first else 0.0,
    }


def citation_metrics(case: Case, output: RunOutput):
    members = [c for c in output.citations if c in output.passages]
    return {
        "status_match": float(output.status == case.expected_status) if output.status else None,
        "citation_membership": len(members) / len(output.citations)
        if output.citations
        else None,
        "evidence_citation_coverage": (
            sum(any(matches(c, e) for c in members) for e in case.expected_evidence)
            / len(case.expected_evidence)
            if case.expected_evidence and output.status
            else None
        ),
    }


def generation_metrics(judgment: Judgment, output: RunOutput, *, fact_count: int):
    if output.status is None:
        raise ValueError("Generation metrics require an accepted final answer.")
    if sorted(f.index for f in judgment.facts) != list(range(fact_count)):
        raise ValueError("Judge must label every required fact exactly once.")
    claims = judgment.claims
    # An empty answer is not perfectly precise; nor does an honest refusal have
    # factual claims to score. N/A stays null and is excluded from means.
    precision = (
        sum(c.correctness == "correct" for c in claims) / len(claims)
        if claims
        else (0.0 if fact_count else None)
    )
    recall = sum(f.status == "met" for f in judgment.facts) / fact_count if fact_count else None
    f1 = None
    if precision is not None and recall is not None:
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "factual_precision": precision,
        "factual_recall": recall,
        "factual_correctness": f1,
        "faithfulness": sum(c.grounded for c in claims) / len(claims) if claims else None,
        "citation_support": sum(c.grounded and c.cited for c in claims) / len(claims)
        if claims
        else None,
    }
