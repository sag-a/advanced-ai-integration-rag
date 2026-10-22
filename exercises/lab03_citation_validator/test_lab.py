import pytest

EXPIRY = "3.3 Οι «Καρδιές» έχουν διάρκεια ισχύος 1 έτος. Η λήξη τους πραγματοποιείται την 1η ημέρα του 13ου μήνα, στις 12:00 μ.μ."
PLAN = "The monthly fee is 35,375 € (Net price before taxes) or 46,05 € (Final price). 500 free minutes."


def test_inline_citations_finds_single_and_grouped_markers_in_order(lab):
    assert lab.inline_citations("Valid for one year [2]. Expires monthly [1, 3] and [2].") == [2, 1, 3]


def test_years_and_other_bracketed_text_are_not_citations(lab):
    assert lab.inline_citations("Prices as of [2024] and [see note] apply [1].") == [1]


def test_valid_answer_returns_cited_numbers(lab):
    assert lab.validate("Hearts last one year [1].", [1], "answered", {1, 2}) == [1]
    assert lab.validate("Hearts last one year.", [2], "answered", {1, 2}) == [2]


def test_the_2024_bug_is_fixed(lab):
    # rag_core.answers.format_answer reads "[2024]" as source 2024 and rejects the turn.
    assert lab.validate("The 2019 list [1] was replaced in [2024].", [], "answered", {1}) == [1]


@pytest.mark.parametrize(
    "text, sources, status, available, reason",
    [
        ("   ", [1], "answered", {1}, "empty"),
        ("Answer [1].", [1, 1], "answered", {1}, "duplicate"),
        ("Answer [7].", [], "answered", {1, 2}, "unknown"),
        ("Answer with no sources.", [], "answered", {1}, "at least one"),
        ("Which program do you mean? [1]", [], "clarification_needed", {1}, "clarification"),
    ],
)
def test_invalid_answers_are_rejected_with_a_reason(lab, text, sources, status, available, reason):
    with pytest.raises(lab.CitationError, match=reason):
        lab.validate(text, sources, status, available)


def test_refusals_may_cite_the_passage_that_shows_the_gap(lab):
    assert lab.validate("Account balances are not in the terms [1].", [], "insufficient_evidence", {1}) == [1]
    assert lab.validate("Not in the documents.", [], "insufficient_evidence", {1}) == []


def test_strip_citations_keeps_years_and_tidies_punctuation(lab):
    assert lab.strip_citations("Valid for one year [1]. Prices from [2024] apply [1, 2].") == (
        "Valid for one year. Prices from [2024] apply."
    )


def test_numbers_must_appear_in_the_cited_passages(lab):
    assert lab.unsupported_numbers("It costs 46.05 € per month with 500 minutes [1].", [PLAN]) == []
    assert lab.unsupported_numbers("It costs 49,90 € per month [1].", [PLAN]) == ["49,90"]


def test_the_exactly_twelve_months_mistake_is_caught(lab):
    # From the course's measured runs: a correct citation of §3.3 with a wrong gloss.
    assert lab.unsupported_numbers("Hearts expire after exactly 12 months [1].", [EXPIRY]) == ["12"]
    assert lab.unsupported_numbers("They expire on the 1st day of the 13th month, at 12:00 [1].", [EXPIRY]) == []
