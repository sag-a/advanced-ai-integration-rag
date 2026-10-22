"""Lab 4: choose a gate threshold on dev data, report it on held-out data. Fill in the TODOs.

A row is {"answerable": bool, "gate_probability": float, ...}. The gate lets a
question through to the LLM when its probability is at least the threshold.
"""


def errors(rows, threshold):
    """{"false_refusals": answerable rows below the threshold,
        "false_answers": unanswerable rows at or above it}"""
    # TODO 1
    raise NotImplementedError


def cost(rows, threshold, refusal_cost=1.0, answer_cost=5.0):
    """Weighted cost: by default a wrong answer is five times worse than a refusal."""
    # TODO 2
    raise NotImplementedError


def candidate_thresholds(rows):
    """[0.0, every midpoint between neighbouring distinct probabilities (rounded to 4
    decimals), 1.01], in increasing order.

    Why midpoints: a threshold equal to an observed score would flip on the smallest
    change in that score.
    """
    # TODO 3
    raise NotImplementedError


def choose_threshold(rows, refusal_cost=1.0, answer_cost=5.0):
    """The candidate with the lowest cost; on a tie, the lowest threshold (fewer refusals)."""
    # TODO 4
    raise NotImplementedError


def rates(rows, threshold):
    """{"false_refusal_rate": false refusals / answerable rows,
        "false_answer_rate": false answers / unanswerable rows}; 0.0 when a group is empty."""
    # TODO 5
    raise NotImplementedError
