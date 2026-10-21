"""A Deep Agents research loop returns evidence to the existing RAG graph."""

import time
from uuid import uuid4
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from deepagents.backends.utils import create_file_data
from langgraph.config import get_stream_writer
from langchain_core.messages import HumanMessage

from rag_core.config import AppError
from rag_core.corpus import digest
from rag_core.retrieval import RetrievalResult
from rag_core.store import retrieve, read_documents
from .guard import ResearchGuard
from .options import options
from .tools import Research
from .prompts import research_prompts, skill_files
from rag_core.templating import bundle_identity



def build_agent(model, research):
    # https://docs.langchain.com/oss/python/deepagents/skills
    # StateBackend keeps skills/scratch virtual: no local filesystem or shell.
    rendered = research_prompts(research.prompt_settings, runtime=True)
    tools = research.tools()
    # Override the SDK default helper so no unconstrained worker is added.
    helper = {
        "name": "general-purpose",
        "description": "Constrained PDF researcher: research one focused corpus topic and return original chunk IDs. No nested delegation or final selection.",
        "system_prompt": rendered["helper"].text,
        "tools": [t for t in tools if t.name != "select_evidence"],
        "middleware": [ResearchGuard(research, helper=True)],
        "skills": ["/skills/"],
    }
    agent = create_deep_agent(
        model=model,
        tools=tools,
        system_prompt=rendered["research"].text,
        skills=["/skills/"],
        backend=StateBackend(),
        subagents=[helper],
        middleware=[ResearchGuard(research)],
        name="magenta-agentic-research",
    )
    agent.step_timeout = research.settings.deadline_seconds
    return agent


class AgenticRetriever:
    def __init__(self, collection, settings, model, *, reporter=None, factory=build_agent):
        self.collection, self.settings, self.model = collection, settings, model
        self.documents = read_documents(collection)
        self.reporter, self.factory = reporter, factory
        budget = options(settings)
        self.configuration = {
            "architecture": "agentic_rag",
            "framework": "deepagents==0.7.19",
            **bundle_identity(research_prompts(settings)),
            "effective_prompt_identity": bundle_identity(research_prompts(settings, runtime=True)),
            "skills_id": digest(skill_files()),
            "max_searches": budget.max_searches,
            "max_clause_reads": budget.max_clause_reads,
            "max_model_calls": budget.max_model_calls,
            "max_tool_calls": budget.max_tool_calls,
            "max_delegations": budget.max_delegations,
            "deadline_seconds": budget.deadline_seconds,
        }

    def __call__(self, question):
        started = time.perf_counter()
        request_id = uuid4().hex[:8]
        try:
            writer = get_stream_writer()
        except RuntimeError:
            writer = None

        def emit(event):
            if writer:
                writer(
                    {
                        "type": "progress",
                        "data": "Agentic research · " + event["event"].replace("_", " "),
                    }
                )
            if self.reporter:
                try:
                    self.reporter({"request_id": request_id, **event})
                except Exception:
                    import logging

                    logging.getLogger(__name__).warning(
                        "Research presentation unavailable; evidence gathering continues."
                    )

        research = Research(
            self.settings,
            lambda q: retrieve(self.collection, self.settings, q),
            self.documents,
            emit,
        )
        agent = self.factory(self.model, research)
        initial = {
            "messages": [{"role": "user", "content": question}],
            "files": {
                path: create_file_data(content) for path, content in skill_files().items()
            },
        }
        try:
            state = agent.invoke(initial, config={"recursion_limit": 70})
            research.check()
            if research.selected is None:
                # One bounded finalization request, not a replay of failed tools.
                # Unsupported requests still need an explicit empty selection.
                research.event("selection_required")
                continuation = {
                    **state,
                    "messages": [
                        *state.get("messages", initial["messages"]),
                        HumanMessage(
                            content="Research is not finalized. You must call select_evidence now using only IDs already retrieved, or [] if the supplied PDF cannot answer. Do not answer the user, invent evidence, or perform further research."
                        ),
                    ],
                    "files": state.get("files", initial["files"]),
                }
                agent.invoke(continuation, config={"recursion_limit": 70})
                research.check()
            if research.selected is None:
                raise AppError(
                    "Agent research ended without selecting evidence. Retry explicitly; no answer was accepted."
                )
        except AppError:
            raise
        except Exception as error:
            raise AppError(
                f"Agentic research failed ({type(error).__name__}); no answer was accepted."
            ) from error
        docs = [research.found[i] for i in research.selected]
        details = {
            **self.configuration,
            "retrieval_id": request_id,
            "events": research.events,
            "searches": research.searches,
            "clause_reads": research.reads,
            "model_calls": research.model_calls,
            "tool_calls": research.tool_calls,
            "delegations": research.delegations,
            "skills_read": [e["path"] for e in research.events if e["event"] == "skill_read"],
            "selected_ids": research.selected,
            "research_usage": research.usage,
        }
        return RetrievalResult(docs, time.perf_counter() - started, details)
