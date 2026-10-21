from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from rag_core.callbacks import ProcessCallback, describe, report_stage
from rag_core.redact import scrub
from rag_core.retrieval import RetrievalResult
from tests.core.test_conversation import Chat, answer, make_graph, passage, turn


def record():
    steps = []
    return steps, ProcessCallback(steps.append)


def test_callback_reports_each_node_start_and_end():
    steps, callback = record()
    state = turn(make_graph(Chat(answer())), "Question", callbacks=[callback])
    assert state["answer_status"] == "answered"
    nodes = [(s["label"], "seconds" in s) for s in steps if s["kind"] == "node"]
    assert nodes == [
        ("prepare_query", False),
        ("prepare_query", True),
        ("retrieve", False),
        ("retrieve", True),
        ("generate_answer", False),
        ("generate_answer", True),
    ]
    assert all(s["depth"] == 0 for s in steps if s["kind"] == "node")


def test_retriever_stage_appears_under_the_retrieve_node():
    def retrieve(query):
        report_stage("vector-seeds", "1 seed passage")
        return RetrievalResult([passage()], 0)

    steps, callback = record()
    turn(make_graph(Chat(answer()), retrieve), "Question", callbacks=[callback])
    stage = next(s for s in steps if s["kind"] == "stage")
    assert stage["label"] == "vector-seeds" and stage["detail"] == "1 seed passage"
    assert stage["depth"] == 1


def test_report_stage_outside_a_run_is_a_no_op():
    report_stage("vector-seeds", "called directly, as by ./rag search")


def test_model_calls_report_model_name_and_tokens_nested_under_their_caller():
    model = FakeListChatModel(responses=["fine"])
    steps, callback = record()
    RunnableLambda(lambda q: model.invoke([HumanMessage(content=q)])).invoke(
        "Question", {"callbacks": [callback]}
    )
    kinds = [s["kind"] for s in steps]
    assert kinds == ["model", "model"]
    assert "seconds" in steps[1]


def test_describe_indents_and_formats_finished_steps():
    line = describe(
        {"kind": "model", "label": "model call", "detail": "12 in / 3 out tokens",
         "depth": 1, "seconds": 0.5}
    )
    assert line == "  ✦ model call done in 0.50s · 12 in / 3 out tokens"
    assert describe({"kind": "node", "label": "retrieve", "detail": "", "depth": 0}) == "◆ retrieve"


def test_scrub_redacts_credentials_and_bounds_history():
    data = {
        "api_key": "private",
        "authorization": "Bearer private",
        "messages": [HumanMessage(content="a" * 3000)]
        + [HumanMessage(content=f"Recent {n}") for n in range(5)]
        + [HumanMessage(content="Current")],
        "text": "secret-example sk-ant-example-credential",
        "metadata": {"checkpoint_ns": "internal"},
    }
    cleaned = scrub(data, ("secret-example",))
    assert "private" not in str(cleaned)
    assert "secret-example" not in str(cleaned)
    assert "sk-ant-example-credential" not in str(cleaned)
    assert "a" * 3000 not in str(cleaned)
    assert len(cleaned["messages"]) == 6
    assert "Current" in str(cleaned)
    assert "checkpoint_ns" not in str(cleaned)
