import pytest

# A tiny fake encoder: each known question maps to a fixed 3-d vector (not normalized).
VECTORS = {
    "When do Hearts expire?": [10, 0, 0],
    "Hearts expiry date?": [9, 1, 0],
    "Plan with 420 minutes?": [0, 10, 0],
    "Plan with 500 minutes?": [0, 10, 0.5],
    "Totally different": [0, 0, 1],
}


def embed(texts):
    return [VECTORS[t] for t in texts]


def test_normalize_and_cosine(lab):
    assert lab.normalize([3, 4]) == pytest.approx([0.6, 0.8])
    assert lab.cosine(lab.normalize([1, 0]), lab.normalize([0, 1])) == pytest.approx(0)
    with pytest.raises(ValueError):
        lab.normalize([0, 0])


def test_an_empty_cache_misses(lab):
    assert lab.SemanticCache(embed).lookup("When do Hearts expire?") is None


def test_a_paraphrase_above_the_threshold_reuses_the_answer(lab):
    cache = lab.SemanticCache(embed, threshold=0.9)
    cache.add("When do Hearts expire?", "1 year, 13th month")
    hit = cache.lookup("Hearts expiry date?")
    assert hit["answer"] == "1 year, 13th month"
    assert hit["cached_question"] == "When do Hearts expire?"
    assert hit["similarity"] == pytest.approx(0.9939, abs=1e-4)


def test_below_the_threshold_misses(lab):
    cache = lab.SemanticCache(embed, threshold=0.999)
    cache.add("When do Hearts expire?", "answer")
    assert cache.lookup("Hearts expiry date?") is None


def test_only_validated_answers_are_stored(lab):
    cache = lab.SemanticCache(embed, threshold=0.5)
    assert cache.add("When do Hearts expire?", "not in the documents", status="insufficient_evidence") is False
    assert cache.lookup("When do Hearts expire?") is None


def test_the_best_match_wins(lab):
    cache = lab.SemanticCache(embed, threshold=0.0)
    cache.add("Totally different", "other")
    cache.add("When do Hearts expire?", "expiry")
    assert cache.lookup("Hearts expiry date?")["answer"] == "expiry"


def test_numbers_match(lab):
    assert lab.numbers_match("Plan with 420 minutes?", "Plan with 420 minutes and 24 Mbps?")
    assert not lab.numbers_match("Plan with 420 minutes?", "Plan with 500 minutes?")
    assert lab.numbers_match("What does it cost per month?", "Cost of the 24 S plan?")


def test_the_guard_stops_a_near_identical_question_with_another_number(lab):
    unguarded = lab.SemanticCache(embed, threshold=0.9)
    unguarded.add("Plan with 500 minutes?", "500-minute plan")
    assert unguarded.lookup("Plan with 420 minutes?")["answer"] == "500-minute plan"  # wrong answer served
    guarded = lab.SemanticCache(embed, threshold=0.9, guard=lab.numbers_match)
    guarded.add("Plan with 500 minutes?", "500-minute plan")
    assert guarded.lookup("Plan with 420 minutes?") is None


def test_a_vetoed_best_match_falls_through_to_the_next(lab):
    cache = lab.SemanticCache(embed, threshold=0.9, guard=lab.numbers_match)
    cache.add("Plan with 500 minutes?", "500-minute plan")
    cache.add("Plan with 420 minutes?", "420-minute plan")
    assert cache.lookup("Plan with 420 minutes?")["answer"] == "420-minute plan"
