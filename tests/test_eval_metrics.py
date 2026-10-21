import pytest

from evals.schema import Case, Passage, RunOutput, Judgment
from evals.metrics import retrieval_metrics, generation_metrics


def passage(clause="1.7", source="rules.pdf"):
    return Passage(
        id=source + clause,
        text="Original evidence",
        source=source,
        pdf_sha256="hash",
        page_start=2,
        page_end=2,
        clause=clause,
    )


def case():
    return Case(
        id="two-facts",
        language="el",
        type="multihop",
        split="dev",
        question="A question",
        expected_status="answered",
        expected_evidence=[
            {"document": "rules.pdf", "page": 2, "clause": "1.7"},
            {"document": "rules.pdf", "page": 2, "clause": "1.9"},
        ],
        expected_facts=["Expiry", "Transfer conditions"],
    )


def test_retrieval_measures_evidence_coverage_and_rank_without_duplicate_credit():
    scores = retrieval_metrics(case(), [passage("other"), passage(), passage()])
    assert scores["recall_at_k"] == 0.5
    assert scores["precision_at_k"] == 0.5
    assert scores["reciprocal_rank"] == 0.5
    assert scores["selected_count"] == 2


def test_missing_expected_document_is_not_relevant_even_with_same_clause():
    assert retrieval_metrics(case(), [passage(source="wrong.pdf")])["recall_at_k"] == 0


def test_intentional_empty_abstention_has_no_factual_grounding_or_citation_score():
    output = RunOutput(
        search_query="question", answer="Insufficient evidence", status="insufficient_evidence"
    )
    scores = generation_metrics(Judgment(facts=[], claims=[]), output, fact_count=0)
    assert scores["faithfulness"] is None and scores["citation_support"] is None
    assert scores["factual_correctness"] is None


def test_correctness_recall_grounding_and_citations_are_separate():
    result = RunOutput(
        search_query="q",
        answer="One correct claim. One unsupported claim.",
        status="answered",
        passages=[passage()],
        citations=[passage()],
    )
    judgment = Judgment(
        facts=[{"index": 0, "status": "met"}, {"index": 1, "status": "missing"}],
        claims=[
            {
                "quote": "One correct claim.",
                "correctness": "correct",
                "grounded": True,
                "cited": True,
            },
            {
                "quote": "One unsupported claim.",
                "correctness": "incorrect",
                "grounded": False,
                "cited": False,
            },
        ],
    )
    scores = generation_metrics(judgment, result, fact_count=2)
    assert (
        scores["factual_recall"]
        == scores["factual_precision"]
        == scores["factual_correctness"]
        == 0.5
    )
    assert scores["faithfulness"] == scores["citation_support"] == 0.5
    judgment.facts[1].index = 0
    with pytest.raises(ValueError, match="every required fact"):
        generation_metrics(judgment, result, fact_count=2)


def test_duplicate_ids_preserve_the_first_rank():
    scores = retrieval_metrics(case(), [passage(), passage("irrelevant"), passage()])
    assert scores["reciprocal_rank"] == 1 and scores["selected_count"] == 2
