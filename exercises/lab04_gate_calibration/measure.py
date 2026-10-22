"""Choose the gate threshold on dev, then see what held-out says. Offline: reads gate_scores.json.

    uv run --all-extras python -m exercises.lab04_gate_calibration.measure
"""

import json
from pathlib import Path

from exercises.common import implementation, print_table

SCORES = Path(__file__).with_name("gate_scores.json")


def main():
    lab = implementation("exercises.lab04_gate_calibration")
    rows = json.loads(SCORES.read_text(encoding="utf-8"))
    dev = [r for r in rows if r["split"] == "dev"]
    heldout = [r for r in rows if r["split"] == "heldout"]
    tuned = lab.choose_threshold(dev)
    cheat = lab.choose_threshold(rows)
    table = []
    for label, threshold in (("course default", 0.5), ("tuned on dev", tuned), ("tuned on everything", cheat)):
        for split, data in (("dev", dev), ("held-out", heldout)):
            e = lab.errors(data, threshold)
            table.append({"threshold": f"{label} ({threshold:.3f})", "evaluated on": split,
                          "false refusals": e["false_refusals"], "false answers": e["false_answers"],
                          **lab.rates(data, threshold)})
    print(f"{len(rows)} questions: dev {len(dev)}, held-out {len(heldout)}\n")
    print_table(table, list(table[0]))
    closest = sorted((r for r in rows if not r["answerable"]), key=lambda r: -r["gate_probability"])[:3]
    print("\nUnanswerable questions the gate liked most:")
    for r in closest:
        print(f"  {r['gate_probability']:.2f}  {r['split']:8} {r['question']}")


if __name__ == "__main__":
    main()
