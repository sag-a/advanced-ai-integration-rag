"""Draw both graphs, then run the questions and show which nodes ran in which step.

    uv run --all-extras python -m exercises.ch03_langgraph_basics.run
    LAB_TARGET=solution uv run --all-extras python -m exercises.ch03_langgraph_basics.run
"""

from exercises.common import implementation

QUESTIONS = [
    "I was charged twice and my router shows a red light. What should I do?",
    "Can I get a full refund within 14 days if I already used the service?",
    "What is the customer service phone number?",
]


def trace(app, question):
    """Run the graph and group the nodes by the step they ran in."""
    by_step, final = {}, None
    for chunk in app.stream({"question": question}, stream_mode=["debug", "values"]):
        mode, data = chunk
        if mode == "debug" and data["type"] == "task":
            by_step.setdefault(data["step"], []).append(data["payload"]["name"])
        elif mode == "values":
            final = data
    return by_step, final


def main():
    lab = implementation("exercises.ch03_langgraph_basics")
    print("=== Part 1: the basic graph (paste into https://mermaid.live to see it)\n")
    print(lab.build_basic_graph().get_graph().draw_mermaid())

    try:
        app = lab.build_agent_graph()
    except Exception as error:
        print(f"\n=== Part 2 is not built yet: {type(error).__name__}: {error}")
        return
    print("=== Part 2: the agent graph\n")
    print(app.get_graph().draw_mermaid())
    for question in QUESTIONS:
        print(f"\nQ: {question}")
        by_step, final = trace(app, question)
        for step, nodes in sorted(by_step.items()):
            together = "   ← in parallel" if len(nodes) > 1 else ""
            print(f"  step {step}: {', '.join(sorted(nodes))}{together}")
        print("  answer:\n    " + final["answer"].replace("\n", "\n    "))


if __name__ == "__main__":
    main()
