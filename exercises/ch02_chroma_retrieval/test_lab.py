"""Grade the three Chroma settings in starter.py.

The tests use a tiny word-overlap embedding instead of the Nomic model, so they run
offline in under a second. measure.py runs the same questions with Nomic.
"""

import pytest

from exercises.fakes import WordOverlap


@pytest.fixture
def collection(lab):
    return lab.make_collection(WordOverlap())


def ids(chunks):
    return [chunk["id"] for chunk in chunks]


# Exercise 1: missing information


def test_1_the_rule_and_its_exception_are_both_retrieved(lab, collection):
    found = ids(lab.missing_information(collection))
    assert "refund-rule-v2" in found and "refund-exception-v2" in found, (
        f"retrieved {found}: the answer needs the refund rule AND its exception. "
        "Try a larger N_RESULTS (compare 1, 3 and 5)"
    )


# Exercise 2: wrong category


def test_2_billing_questions_search_billing(lab, collection):
    found = lab.wrong_category(collection)
    categories = {chunk["metadata"]["category"] for chunk in found}
    assert categories == {"billing"}, f"results came from {sorted(categories)}: fix the category in BILLING_FILTER"


def test_2_the_duplicate_charge_policy_is_found(lab, collection):
    assert "duplicate-charge" in ids(lab.wrong_category(collection))


# Exercise 3: outdated information


def test_3_no_outdated_policy_is_returned(lab, collection):
    outdated = [c["id"] for c in lab.outdated_information(collection) if c["metadata"]["status"] != "current"]
    assert not outdated, (
        f"{outdated} is an outdated policy: it says 30 days, the current one says 14. "
        "Add a status or version condition to POLICY_FILTER"
    )


def test_3_the_current_refund_rule_is_returned(lab, collection):
    found = lab.outdated_information(collection)
    assert "refund-rule-v2" in ids(found), "the filter is too strict: the current refund rule is gone"
    assert {c["metadata"]["category"] for c in found} == {"billing"}, "keep the billing category in POLICY_FILTER"


def test_the_policies_are_unchanged(lab):
    # The task is to change the query settings, not the data.
    assert len(lab.POLICIES) == 8
    assert ("refund-rule-v1", "superseded") in {(i, m["status"]) for i, _, m in lab.POLICIES}
