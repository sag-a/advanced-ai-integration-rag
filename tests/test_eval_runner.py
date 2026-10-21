from dataclasses import replace

import pytest
from langchain_core.documents import Document

from evals.adapter import execute
from evals.schema import Question
from rag_core.answers import Answer
from rag_core.config import AppError, Settings
from rag_core.graph import Rewrite
from rag_core.registry import Architecture
from rag_core.retrieval import RetrievalResult
from tests.test_eval_metrics import case


def document():
    return Document(
        id="original",
        page_content="Only original evidence.",
        metadata=dict(
            source="rules.pdf", pdf_sha256="hash", page_start=2, page_end=2, clause="1.7"
        ),
    )


class Chat:
    def __init__(self):
        self.payloads = []

    def call(self, schema, system, payload):
        self.payloads.append(payload)
        if schema is Rewrite:
            assert payload["history"]
            return Rewrite(search_query="Standalone question", clarification=""), dict(
                usage={"input_tokens": 2}
            )
        return Answer(answer="Supported claim", sources=[1], status="answered"), dict(
            usage={"output_tokens": 3}
        )


def plugin(seen):
    def factory(settings, collection, log):
        def retrieve(query):
            seen.append(query)
            return RetrievalResult([document()], 0.1, {"selected_ids": ["original"]})

        return retrieve

    return Architecture(name="fake", label="Fake", make_retriever=factory)


def test_adapter_runs_real_graph_and_history_without_gold_or_classroom_files(
    tmp_path, monkeypatch
):
    seen, chat = [], Chat()
    request = Question(
        id="q",
        question="And then?",
        history=[
            dict(role="human", text="A subject"),
            dict(role="ai", text="A previous answer"),
        ],
    )
    monkeypatch.chdir(tmp_path)
    settings = Settings(architecture="fake")
    output = execute(request, settings, plugin(seen), None, chat)
    assert output.status == "answered" and output.retrieval_complete
    assert seen == ["Standalone question"]
    assert output.citations[0].id == "original"
    assert not list(tmp_path.iterdir())
    assert not any("expected_facts" in p or "expected_status" in p for p in chat.payloads)
    with pytest.raises(TypeError, match="never Case"):
        execute(case(), settings, plugin(seen), None, chat)


def test_no_api_retrieval_excludes_model_research_and_history():
    architecture = replace(plugin([]), needs_llm_for_retrieval=True)
    with pytest.raises(AppError, match="excluded"):
        execute(
            Question(id="q", question="Question"),
            Settings(),
            architecture,
            None,
            retrieval_only=True,
        )
    architecture = replace(architecture, needs_llm_for_retrieval=False)
    with pytest.raises(AppError, match="excluded"):
        execute(
            Question(
                id="q", question="Question", history=[dict(role="human", text="Old question")]
            ),
            Settings(),
            architecture,
            None,
            retrieval_only=True,
        )
    output = execute(
        Question(id="q", question="Question"),
        Settings(),
        architecture,
        None,
        retrieval_only=True,
    )
    assert output.retrieval_complete and not output.answer and output.status is None


def test_generation_failure_preserves_retrieval_without_accepting_an_answer():
    class Fail:
        def call(self, *args):
            raise AppError("Model unavailable")

    output = execute(
        Question(id="q", question="Question"), Settings(), plugin([]), None, Fail()
    )
    assert output.retrieval_complete and output.passages
    assert output.error and not output.answer and not output.citations


def test_comparison_is_sequential_and_judge_failure_is_not_a_perfect_score(monkeypatch):
    from types import SimpleNamespace
    from evals.run import evaluate
    from evals.schema import RunOutput
    from tests.test_eval_metrics import passage

    calls = []
    scenario = case()
    scenario.expected_evidence = scenario.expected_evidence[:1]
    monkeypatch.setattr("evals.run.read_documents", lambda _: [document()])
    monkeypatch.setattr("evals.run.get", lambda name: replace(plugin([]), name=name))

    def run(request, settings, *args, **kwargs):
        assert type(request) is Question
        assert not hasattr(request, "expected_facts")
        calls.append((settings.architecture, request.id))
        return RunOutput(
            search_query=request.question,
            answer="Supported claim",
            status="answered",
            passages=[passage()],
            retrieval_complete=True,
        )

    monkeypatch.setattr("evals.run.execute", run)

    def fail(*args):
        raise ValueError("Bad judge labels")

    report = evaluate(
        Settings(),
        [scenario],
        ["first", "second"],
        None,
        manifest={"dataset_id": "dataset"},
        judge=SimpleNamespace(configuration={}, evaluate=fail),
        progress=lambda *_: None,
    )
    assert calls == [("first", scenario.id), ("second", scenario.id)]
    assert not report["complete"]
    assert all(
        row["judge_error"] and row["metrics"]["factual_correctness"] is None
        for row in report["rows"]
    )


def test_score_export_failure_warns_and_keeps_local_results():
    from types import SimpleNamespace
    from evals.run import publish_scores
    from evals.schema import RunOutput

    def fail(**kwargs):
        raise ConnectionError("Unavailable")

    warning = publish_scores(
        SimpleNamespace(client=SimpleNamespace(create_score=fail)),
        RunOutput(search_query="q", trace_id="trace"),
        {"recall_at_k": 0.5},
        {"run_id": "test"},
    )
    assert "upload failed" in warning


def test_existing_report_is_rejected_before_any_service_access(monkeypatch, tmp_path):
    from evals.run import main

    monkeypatch.setattr("evals.run.connect", lambda *_: pytest.fail("No service call"))
    path = tmp_path / "existing.json"
    path.write_text("{}")
    assert main(["--output", str(path)]) == 1
