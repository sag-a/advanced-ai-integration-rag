"""One explicit Anthropic model for structured rewriting and grounded answers."""

import json
import time

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import convert_to_anthropic_tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.json import parse_partial_json

from .config import AppError


class AnthropicChat:
    def __init__(self, settings, *, max_tokens=1200):
        # Use the project's shell-over-dotenv convention without the existing
        # factory's env-file synchronization side effects or catalog defaults.
        self.model = ChatAnthropic(
            model=settings.model,
            api_key=settings.api_key(),
            timeout=30,
            max_retries=1,
            max_tokens=max_tokens,
            **(
                {"temperature": settings.temperature}
                if settings.temperature is not None
                else {}
            ),
        )

    def stream(self, schema, system, payload, on_draft):
        """Stream answer snapshots, then strictly validate the completed tool call.

        Partial JSON is used for display only. It must never repair final output.
        """
        started = time.perf_counter()
        combined, previous = None, ""
        try:
            # Stream tool arguments before provider-side JSON buffering completes.
            # https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming
            tool = {**convert_to_anthropic_tool(schema), "eager_input_streaming": True}
            bound = self.model.bind_tools([tool], tool_choice=schema.__name__)
            for chunk in bound.stream(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ]
            ):
                combined = chunk if combined is None else combined + chunk
                calls = combined.tool_call_chunks
                if len(calls) == 1 and calls[0].get("name") == schema.__name__:
                    try:
                        partial = parse_partial_json(calls[0]["args"])
                    except (ValueError, TypeError):
                        continue
                    text = partial.get("answer") if isinstance(partial, dict) else None
                    if isinstance(text, str) and text and text != previous:
                        on_draft(text)
                        previous = text
            if combined is None or combined.response_metadata.get("stop_reason") != "tool_use":
                raise AppError(
                    "Anthropic stream was interrupted or truncated; no answer was accepted."
                )
            calls = combined.tool_call_chunks
            if len(calls) != 1 or calls[0].get("name") != schema.__name__:
                raise AppError(
                    "Anthropic returned invalid structured output; no answer was accepted."
                )
            # json.loads deliberately rejects unfinished JSON accepted by the
            # partial parser above, even if it appears to contain all fields.
            parsed = schema.model_validate(json.loads(calls[0]["args"]))
            return parsed, {
                "elapsed_seconds": time.perf_counter() - started,
                "usage": combined.usage_metadata or {},
            }
        except AppError:
            raise
        except Exception as error:
            raise AppError(
                f"Anthropic stream failed ({type(error).__name__}); no answer was accepted."
            ) from error

    def call(self, schema, system, payload):
        started = time.perf_counter()
        try:
            result = self.model.with_structured_output(
                schema, method="function_calling", include_raw=True
            ).invoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
                ]
            )
            if result.get("parsing_error") or not isinstance(result.get("parsed"), schema):
                raise AppError(
                    "Anthropic returned incomplete or invalid structured output; no answer was accepted."
                )
            raw = result["raw"]
            if raw.response_metadata.get("stop_reason") not in ("end_turn", "tool_use"):
                raise AppError(
                    "Anthropic output was interrupted or truncated; no complete answer was accepted."
                )
            return result["parsed"], {
                "elapsed_seconds": time.perf_counter() - started,
                "usage": raw.usage_metadata or {},
            }
        except AppError:
            raise
        except Exception as error:
            raise AppError(
                f"Anthropic request failed ({type(error).__name__}, status={getattr(error, 'status_code', None)}). No answer was accepted."
            ) from error
