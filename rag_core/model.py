"""One explicit Claude model for structured rewriting and grounded answers.

Calls go to OpenRouter (the default) or Anthropic, chosen by LLM_PROVIDER. Both speak
the Anthropic Messages API, so the same ChatAnthropic client serves both."""

import json
import time

from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import convert_to_anthropic_tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.utils.json import parse_partial_json

from .config import AppError

# Sampling parameters (temperature, top_p, top_k) were removed from the Claude API:
# Claude 4.7 and later models reject them with a 400 error. Only these older model
# lines still accept them. https://platform.claude.com/docs/en/about-claude/model-deprecations
ACCEPTS_TEMPERATURE = (
    "claude-haiku-4-5",
    "claude-sonnet-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-5",
    "claude-opus-4-6",
)


def accepts_temperature(model):
    """True when the model still accepts a temperature setting."""
    return model.startswith(ACCEPTS_TEMPERATURE)


class AnthropicChat:
    via = ""  # " via OpenRouter" or " via Anthropic", for error messages

    def __init__(self, settings, *, max_tokens=1200):
        self.via = " via OpenRouter" if settings.provider() == "openrouter" else " via Anthropic"
        self.key_name = settings.api_key_name()
        base_url = settings.base_url()
        # Use the project's shell-over-dotenv convention without the existing
        # factory's env-file synchronization side effects or catalog defaults.
        self.model = ChatAnthropic(
            model=settings.provider_model(),
            api_key=settings.api_key(),
            **({"base_url": base_url} if base_url else {}),
            timeout=30,
            max_retries=1,
            max_tokens=max_tokens,
            **(
                {"temperature": settings.temperature}
                if settings.temperature is not None and accepts_temperature(settings.model)
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
                    f"Claude stream{self.via} was interrupted or truncated; no answer was accepted."
                )
            calls = combined.tool_call_chunks
            if len(calls) != 1 or calls[0].get("name") != schema.__name__:
                raise AppError(
                    f"Claude{self.via} returned invalid structured output; no answer was accepted."
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
                f"Claude stream{self.via} failed ({type(error).__name__}); no answer was accepted."
                + self.key_hint(error)
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
                    f"Claude{self.via} returned incomplete or invalid structured output; no answer was accepted."
                )
            raw = result["raw"]
            if raw.response_metadata.get("stop_reason") not in ("end_turn", "tool_use"):
                raise AppError(
                    f"Claude output{self.via} was interrupted or truncated; no complete answer was accepted."
                )
            return result["parsed"], {
                "elapsed_seconds": time.perf_counter() - started,
                "usage": raw.usage_metadata or {},
            }
        except AppError:
            raise
        except Exception as error:
            raise AppError(
                f"Claude request{self.via} failed ({type(error).__name__}, status={getattr(error, 'status_code', None)}). No answer was accepted."
                + self.key_hint(error)
            ) from error

    def key_hint(self, error):
        """A rejected key is the most common classroom failure: name the key to check."""
        if getattr(error, "status_code", None) == 401 or "Authentication" in type(error).__name__:
            return f" Check {getattr(self, 'key_name', 'the API key')} in .env and restart."
        return ""
