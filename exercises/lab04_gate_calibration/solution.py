"""Reference solution: choose a gate threshold on dev data, report it on held-out data.

A row is {"answerable": bool, "gate_probability": float, ...}. The gate lets a
question through to the LLM when its probability is at least the threshold.
"""


def errors(rows, threshold):
    """Count both kinds of mistake at a threshold."""
    false_refusals = sum(r["answerable"] and r["gate_probability"] < threshold for r in rows)
    false_answers = sum(not r["answerable"] and r["gate_probability"] >= threshold for r in rows)
    return {"false_refusals": false_refusals, "false_answers": false_answers}


def cost(rows, threshold, refusal_cost=1.0, answer_cost=5.0):
    """Weighted cost: by default a wrong answer is five times worse than a refusal."""
    e = errors(rows, threshold)
    return e["false_refusals"] * refusal_cost + e["false_answers"] * answer_cost


def candidate_thresholds(rows):
    """0, 1.01, and every midpoint between neighbouring distinct probabilities.

    Midpoints keep a margin on both sides: a threshold equal to an observed score
    would flip on the smallest change in that score.
    """
    values = sorted({r["gate_probability"] for r in rows})
    midpoints = [round((a + b) / 2, 4) for a, b in zip(values, values[1:])]
    return [0.0, *midpoints, 1.01]


def choose_threshold(rows, refusal_cost=1.0, answer_cost=5.0):
    """The candidate with the lowest cost; on a tie, the lowest threshold (fewer refusals)."""
    return min(candidate_thresholds(rows), key=lambda t: (cost(rows, t, refusal_cost, answer_cost), t))


def rates(rows, threshold):
    """Error rates per kind: false refusals over answerable rows, false answers over the rest."""
    answerable = sum(r["answerable"] for r in rows)
    unanswerable = len(rows) - answerable
    e = errors(rows, threshold)
    return {
        "false_refusal_rate": e["false_refusals"] / answerable if answerable else 0.0,
        "false_answer_rate": e["false_answers"] / unanswerable if unanswerable else 0.0,
    }
