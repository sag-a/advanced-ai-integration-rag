from types import SimpleNamespace

import pytest

from evals.judges import AnthropicJudge
from evals.metrics import generation_metrics
from evals.report import aggregate
from evals.schema import Judgment, RunOutput
from tests.test_eval_metrics import case, passage


def judge(verdict):
    return AnthropicJudge(SimpleNamespace(call=lambda *args: (verdict, {"usage": {}})), "fake")


def output():
    return RunOutput(
        search_query="q",
        answer="A supported claim.",
        status="answered",
        passages=[passage()],
        citations=[passage()],
    )


def verdict():
    return Judgment(
        facts=[
            dict(index=0, status="met", quote="A supported claim."),
            dict(index=1, status="missing"),
        ],
        claims=[
            dict(quote="A supported claim.", correctness="correct", grounded=True, cited=True)
        ],
    )


def test_judge_checks_fact_completeness_quotes_and_actual_citation_presence():
    assert judge(verdict()).evaluate(case(), output(), [passage()])[0].claims
    bad = verdict()
    bad.claims[0].quote = "An invented answer claim"
    with pytest.raises(ValueError, match="verbatim"):
        judge(bad).evaluate(case(), output(), [passage()])
    uncited = output()
    uncited.citations = []
    with pytest.raises(ValueError, match="absent"):
        judge(verdict()).evaluate(case(), uncited, [passage()])
    bad = verdict()
    bad.facts = []
    with pytest.raises(ValueError, match="every required fact"):
        judge(bad).evaluate(case(), output(), [passage()])


def test_empty_claims_do_not_inflate_answerable_or_abstention_scores():
    no_claims = verdict()
    no_claims.claims = []
    scores = generation_metrics(no_claims, output(), fact_count=2)
    assert scores["factual_correctness"] == 0 and scores["faithfulness"] is None
    with pytest.raises(ValueError, match="omitted"):
        judge(no_claims).evaluate(case(), output(), [passage()])


def test_null_metrics_have_their_own_denominators_and_failures_remain_visible():
    rows = [
        dict(
            architecture="fake",
            type="factual",
            language="el",
            outcome="completed",
            metrics={"faithfulness": None, "recall_at_k": 0.5},
        ),
        dict(
            architecture="fake",
            type="factual",
            language="el",
            outcome="failed",
            metrics={"faithfulness": None},
        ),
    ]
    group = aggregate(rows)[0]
    assert group["metrics"]["faithfulness"] == dict(mean=None, n=0)
    assert group["metrics"]["recall_at_k"] == dict(mean=0.5, n=1)
    assert group["completed"] == 1 and group["failed"] == 1


def test_fixed_refusal_uses_no_model_and_never_claims_reference_facts_were_answered():
    from rag_core.answers import insufficient

    class NoModel:
        def call(self, *args):
            pytest.fail("A canonical refusal needs no judge call")

    result = output()
    result.status = "insufficient_evidence"
    result.answer = insufficient("A question") + "\n[1] rules.pdf · PDF p. 2 · §1.7"
    labels, usage = AnthropicJudge(NoModel(), "fake").evaluate(case(), result, [passage()])
    assert all(f.status == "missing" for f in labels.facts) and labels.claims == []
    scores = generation_metrics(labels, result, fact_count=2)
    assert scores["factual_recall"] == 0 and scores["factual_correctness"] == 0
    assert scores["faithfulness"] is None and scores["citation_support"] is None
    assert usage["completed_calls"] == 0


def test_fact_coverage_requires_a_quote_from_the_answer_not_the_source_footer():
    labels = verdict()
    labels.facts[0].quote = "rules.pdf"
    result = output()
    result.answer += "\n[1] rules.pdf · PDF p. 2 · §1.7"
    with pytest.raises(ValueError, match="verbatim answer quote"):
        judge(labels).evaluate(case(), result, [passage()])


def test_invalid_judge_labels_get_one_bounded_retry_with_both_usages_retained():
    labels = verdict()
    labels.facts[0].quote = "A quote from the reference instead of the answer"
    calls = []

    class RetryJudge:
        def call(self, *args):
            calls.append(args[2].copy())
            return (labels if len(calls) == 1 else verdict()), dict(
                usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
            )

    result, usage = AnthropicJudge(RetryJudge(), "fake").evaluate(case(), output(), [passage()])
    assert result.facts[0].quote == "A supported claim."
    assert len(calls) == 2 and "validation_feedback" in calls[1]
    assert usage["completed_calls"] == 2 and usage["usage"]["input_tokens"] == 20
    assert (
        usage["attempts"][0]["validation_errors"]
        and not usage["attempts"][1]["validation_errors"]
    )
