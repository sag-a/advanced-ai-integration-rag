import pytest


def row(answerable, p):
    return {"answerable": answerable, "gate_probability": p}


ROWS = [row(True, 0.95), row(True, 0.80), row(True, 0.60), row(False, 0.55), row(False, 0.20)]


def test_errors_counts_both_kinds_of_mistake(lab):
    assert lab.errors(ROWS, 0.5) == {"false_refusals": 0, "false_answers": 1}
    assert lab.errors(ROWS, 0.7) == {"false_refusals": 1, "false_answers": 0}
    assert lab.errors(ROWS, 0.6) == {"false_refusals": 0, "false_answers": 0}  # >= lets 0.60 through


def test_cost_weights_wrong_answers_more_than_refusals(lab):
    assert lab.cost(ROWS, 0.5) == 5.0
    assert lab.cost(ROWS, 0.7) == 1.0
    assert lab.cost(ROWS, 0.5, refusal_cost=1, answer_cost=1) == 1.0


def test_candidates_are_midpoints_plus_the_extremes(lab):
    assert lab.candidate_thresholds([row(True, 0.8), row(False, 0.2), row(True, 0.8)]) == [0.0, 0.5, 1.01]


def test_choose_threshold_finds_the_gap_between_the_classes(lab):
    assert lab.choose_threshold(ROWS) == 0.575  # between 0.55 and 0.60: no errors


def test_the_cost_ratio_moves_the_threshold(lab):
    overlap = [row(True, 0.9), row(True, 0.4), row(False, 0.6), row(False, 0.1)]
    strict = lab.choose_threshold(overlap, refusal_cost=1, answer_cost=5)
    lenient = lab.choose_threshold(overlap, refusal_cost=5, answer_cost=1)
    assert strict > 0.6 and lenient < 0.4


def test_ties_prefer_the_lower_threshold(lab):
    assert lab.choose_threshold([row(True, 0.9), row(False, 0.1)]) == 0.5


def test_rates_divide_by_the_right_population(lab):
    r = lab.rates(ROWS, 0.7)
    assert r["false_refusal_rate"] == pytest.approx(1 / 3)
    assert r["false_answer_rate"] == 0.0
    assert lab.rates([row(True, 0.9)], 0.5) == {"false_refusal_rate": 0.0, "false_answer_rate": 0.0}
