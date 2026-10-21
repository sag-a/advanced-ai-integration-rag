"""START -> prepare_query -> retrieve -> generate_answer -> END."""

from typing import Annotated, TypedDict
from contextlib import closing
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from .config import AppError
from .answers import Answer, format_answer, insufficient
from .retrieval import RetrievalResult
from . import prompts
from .templating import bundle_identity


class Rewrite(BaseModel):
    model_config = ConfigDict(extra="forbid")
    search_query: str = Field(
        description="Standalone full question; empty when its referent is ambiguous. Never expand an ambiguous pronoun to all possible subjects."
    )
    clarification: str = Field(
        description="Ask which subject is meant when multiple referents are plausible; otherwise empty."
    )


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    search_query: str
    documents: list[Document]
    answer_status: str
    clarification: str
    source_map: list[dict]
    observation: dict


HISTORY_MESSAGE_LIMIT = 6


def history_view(messages):
    """Show the last six messages; leave the caller's full message list intact."""
    return [
        {"role": message.type, "text": str(message.content)}
        for message in messages[-HISTORY_MESSAGE_LIMIT:]
    ]


def build_graph(chat, retrieve_documents, settings, *, entry=None):
    rendered = prompts.for_settings(settings)
    prompt_identity = bundle_identity(rendered)

    # https://docs.langchain.com/oss/python/langgraph/graph-api
    def prepare_query(state):
        get_stream_writer()({"type": "progress", "data": "Preparing question"})
        current = state["messages"][-1]
        update = {
            "question": current.content,
            "search_query": current.content,
            "documents": [],
            "source_map": [],
            "answer_status": "unset",
            "clarification": "",
            "observation": dict(prompt_identity),
        }
        if len(state["messages"]) == 1:
            return update

        previous_messages = state["messages"][:-1]
        history = history_view(previous_messages)
        rewrite, usage = chat.call(
            Rewrite,
            rendered["rewrite"].text,
            {
                "history": history,
                "history_truncated": len(history) < len(previous_messages),
                "question": current.content,
            },
        )
        if bool(rewrite.search_query.strip()) == bool(rewrite.clarification.strip()):
            raise AppError(
                "Query rewrite must provide either a search query or a clarification."
            )
        update.update(
            search_query=rewrite.search_query.strip(),
            clarification=rewrite.clarification.strip(),
            observation={**prompt_identity, "rewrite": usage},
        )
        if rewrite.clarification:
            update["answer_status"] = "clarification_needed"
        return update

    def retrieve(state):
        get_stream_writer()({"type": "progress", "data": "Searching PDF"})
        if state["answer_status"] == "clarification_needed":
            return {"documents": [], "source_map": []}
        result = retrieve_documents(state["search_query"])
        if not isinstance(result, RetrievalResult):
            raise AppError("Retriever must return a RetrievalResult.")
        sources = [
            {
                "number": number,
                "id": document.id,
                "page_start": document.metadata["page_start"],
                "page_end": document.metadata["page_end"],
                "clause": document.metadata.get("clause", ""),
                "source": document.metadata["source"],
                "text": document.page_content,
            }
            for number, document in enumerate(result.documents, 1)
        ]
        observations = {
            **state["observation"],
            "retrieval_seconds": result.elapsed_seconds,
        }
        if result.details:
            observations["retrieval_details"] = result.details
        return {
            "documents": result.documents,
            "source_map": sources,
            "observation": observations,
        }

    def generate_answer(state, config: RunnableConfig):
        writer = get_stream_writer()
        writer({"type": "progress", "data": "Generating answer"})
        observations = state["observation"]
        if state["answer_status"] == "clarification_needed":
            answer = Answer(
                answer=state["clarification"], sources=[], status="clarification_needed"
            )
        elif not state["documents"]:
            answer = Answer(
                answer=insufficient(state["question"]),
                sources=[],
                status="insufficient_evidence",
            )
        else:
            evidence = [
                {**source, "text": doc.page_content}
                for source, doc in zip(state["source_map"], state["documents"], strict=True)
            ]
            payload = {
                "question": state["question"],
                "search_query": state["search_query"],
                "history": history_view(state["messages"][:-1]),
                "evidence": evidence,
            }
            if config.get("configurable", {}).get("stream_answer", False):
                answer, usage = chat.stream(
                    Answer,
                    rendered["answer"].text,
                    payload,
                    on_draft=lambda text: writer({"type": "draft", "data": text}),
                )
            else:
                answer, usage = chat.call(Answer, rendered["answer"].text, payload)
            observations = {**observations, "generation": usage}
        text, cited = format_answer(answer, state["source_map"], state["question"])
        summary = {
            "answer_status": answer.status,
            "source_map": cited,
            "search_query": state["search_query"],
            "question": state["question"],
            "observation": observations,
        }
        message = AIMessage(
            id="answer:" + state["messages"][-1].id,
            content=text,
            additional_kwargs={"result": summary},
        )
        return {
            "messages": [message],
            "answer_status": answer.status,
            "observation": observations,
        }

    builder = StateGraph(State)
    builder.add_node("prepare_query", prepare_query)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate_answer", generate_answer)
    if entry is None:
        builder.add_edge(START, "prepare_query")
    else:
        # An entry hook returns a LangGraph Command to continue or finish.
        builder.add_node("entry", entry, destinations=("prepare_query", END))
        builder.add_edge(START, "entry")
    builder.add_edge("prepare_query", "retrieve")
    builder.add_edge("retrieve", "generate_answer")
    builder.add_edge("generate_answer", END)
    graph = builder.compile()
    # A retrieval node may do bounded tool research. Allow its deadline plus cleanup.
    graph.step_timeout = settings.extras.get("deadline_seconds", 150) + 30
    return graph


def run_turn(
    graph,
    question,
    messages,
    settings,
    *,
    streaming=False,
    message_id=None,
    callbacks=None,
):
    """Pass a message list to the graph; callers retain history only on final.

    There is no saved state or conversation database. State events also let the
    evaluator inspect completed retrieval when later answer generation fails.
    ``callbacks`` are LangChain callback handlers, e.g. ``ProcessCallback``; LangGraph
    passes them to every node, model call and tool call in the turn.
    """
    if not question.strip() or len(question.encode("utf-8")) > 4000:
        raise AppError("Use a nonempty question up to 4000 UTF-8 bytes.")
    message_id = message_id or str(uuid4())
    inputs = {"messages": [*messages, HumanMessage(id=message_id, content=question)]}
    config = {
        "configurable": {"stream_answer": streaming},
        "recursion_limit": 10,
        "callbacks": list(callbacks or []),
    }
    result = None
    try:
        with closing(
            graph.stream(inputs, config, stream_mode=["custom", "values"], version="v2")
        ) as events:
            for event in events:
                if event["type"] == "values":
                    result = event["data"]
                    yield {"type": "state", "data": result}
                elif streaming:
                    yield event["data"]
        if not result or result["messages"][-1].type != "ai":
            raise AppError("The graph ended without a validated assistant answer.")
        yield {"type": "final", "data": result}
    except AppError:
        raise
    except Exception as error:
        raise AppError(
            f"Turn failed ({type(error).__name__}); no answer was accepted."
        ) from error
