"""Optional Langfuse tracing. Credentials never enter graph state or telemetry."""

from contextlib import contextmanager
from functools import lru_cache
import base64
import sys
import re
from types import SimpleNamespace
from urllib.parse import urlsplit

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from .embeddings import CONTRACT
from . import prompts
from .templating import bundle_identity
from .config import AppError
from .corpus import corpus_identity
from .graph import HISTORY_MESSAGE_LIMIT, history_view


WARNING = "Langfuse export failed or is unavailable. The RAG answer remains valid; the trace may be incomplete."


def credentials(settings):
    config = {
        "public_key": settings.env("LANGFUSE_PUBLIC_KEY", required=True),
        "secret_key": settings.env("LANGFUSE_SECRET_KEY", required=True),
        "base_url": settings.env("LANGFUSE_HOST", required=True),
    }
    parsed = urlsplit(config["base_url"])
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise AppError(
            "LANGFUSE_HOST must be an HTTPS service URL without credentials or query parameters."
        )
    config["base_url"] = config["base_url"].rstrip("/")
    return config


def scrub(value, secrets=()):
    """Keep prompt-sized history; remove internal state and known credentials."""
    if isinstance(value, BaseMessage):
        return {"role": value.type, "text": scrub(value.content, secrets)}
    if isinstance(value, Document):
        return {
            "id": value.id,
            "text": scrub(value.page_content, secrets),
            **scrub(value.metadata, secrets),
        }
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            label = str(key).lower()
            if re.search(
                r"api.?key|secret|password|authorization|cookie|token$", label
            ) and not label.endswith("_tokens"):
                clean[key] = "[REDACTED]"
            elif label.startswith(("checkpoint", "__pregel")) or label in {
                "configurable",
                "callbacks",
            }:
                continue
            elif (
                key == "messages"
                and isinstance(item, list)
                and all(isinstance(m, BaseMessage) for m in item)
            ):
                clean[key] = scrub(history_view(item), secrets)
            else:
                clean[key] = scrub(item, secrets)
        return clean
    if isinstance(value, (list, tuple)):
        return [scrub(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        return re.sub(r"\b(?:sk-|pk-lf-)[A-Za-z0-9_-]+", "[REDACTED]", value)
    return value


def trace_metadata(settings):
    return {
        "architecture": settings.architecture,
        "collection": settings.collection,
        "top_k": settings.top_k,
        "model": settings.model,
        "corpus_id": corpus_identity(settings.corpus),
        "embedding_contract": CONTRACT,
        **bundle_identity(prompts.for_settings(settings)),
        "history_message_limit": HISTORY_MESSAGE_LIMIT,
        "temperature": settings.temperature,
    }


@contextmanager
def optional_trace(telemetry, *args):
    """Telemetry failures cannot turn a successful graph request into a failure."""
    state = SimpleNamespace(trace=None, warning=None)
    context = None
    if telemetry:
        try:
            context = telemetry.turn(*args)
            state.trace = context.__enter__()
        except Exception:
            state.warning = "Langfuse tracing is unavailable; this turn may have no trace."
            context = None
    try:
        yield state
    except BaseException:
        if context:
            try:
                context.__exit__(*sys.exc_info())
            except BaseException:
                pass  # Preserve the original application error.
        raise
    else:
        if context:
            try:
                context.__exit__(None, None, None)
            except Exception:
                state.warning = "Langfuse trace finalization failed; the answer remains valid."


@lru_cache(maxsize=4)
def _cloud_resources(public_key, secret_key, base_url, provider_key):
    try:
        from langfuse import Langfuse
        from langfuse.types import MaskOtelSpansResult, OtelSpanPatch
        from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
    except ImportError as error:
        raise AppError("Install the optional --extra langfuse dependency.") from error
    config = {"public_key": public_key, "secret_key": secret_key, "base_url": base_url}
    # Include the provider key only for masking; do not send it to Langfuse.
    secrets = (
        config["secret_key"],
        config["public_key"],
        provider_key,
    )

    def mask_spans(*, params):
        patches = {}
        for identifier, span in params.spans.items():
            attributes = {}
            for key, value in span.attributes.items():
                if "exception" in key or key.endswith("status_message"):
                    attributes[key] = (
                        "Operation failed; inspect the sanitized application error."
                    )
                elif isinstance(value, str):
                    attributes[key] = scrub(value, secrets)
            patches[identifier] = OtelSpanPatch(set_attributes=attributes)
        return MaskOtelSpansResult(span_patches=patches)

    class CheckedExporter(SpanExporter):
        def __init__(self):
            auth = base64.b64encode(
                f"{config['public_key']}:{config['secret_key']}".encode()
            ).decode()
            self.delegate = OTLPSpanExporter(
                endpoint=config["base_url"] + "/api/public/otel/v1/traces",
                headers={
                    "Authorization": "Basic " + auth,
                    "x-langfuse-ingestion-version": "4",
                },
                timeout=10,
            )
            self.failures = 0

        def export(self, spans):
            try:
                result = self.delegate.export(spans)
            except Exception:
                result = SpanExportResult.FAILURE
            if result != SpanExportResult.SUCCESS:
                self.failures += 1
            return result

        def shutdown(self):
            self.delegate.shutdown()

    exporter = CheckedExporter()
    client = Langfuse(
        **config,
        timeout=10,
        debug=False,
        tracing_enabled=True,
        environment="rag-course",
        mask=lambda data: scrub(data, secrets),
        mask_otel_spans=mask_spans,
        span_exporter=exporter,
    )

    return client, exporter


class Telemetry:
    def __init__(self, settings, *, client=None, public_key=None):
        self.settings = settings
        self.exporter = None
        self._seen_failures = 0
        if client is not None:
            if not public_key:
                raise AppError("An injected Langfuse client requires its public key.")
            self.client = client
            self.public_key = public_key
            return
        config = credentials(settings)
        self.public_key = config["public_key"]
        provider_key = settings.env("ANTHROPIC_API_KEY")
        self.client, self.exporter = _cloud_resources(**config, provider_key=provider_key)
        self._seen_failures = self.exporter.failures

    def check(self):
        try:
            if not self.client.auth_check():
                raise AppError(
                    "Langfuse authentication failed. Check project keys and region URL."
                )
            return {
                "authenticated": True,
                "project_url": self.client.get_trace_url(trace_id="0" * 32).split("/traces/")[
                    0
                ],
            }
        except AppError:
            raise
        except Exception as error:
            raise AppError(
                "Cannot authenticate with Langfuse. Check connectivity, project keys, and region URL."
            ) from error

    @contextmanager
    def turn(self, question, history, thread_id, message_id):
        from langfuse import propagate_attributes
        from langfuse.langchain import CallbackHandler

        class BoundedCallback(CallbackHandler):
            def on_chain_start(self, serialized, inputs, **kwargs):
                return super().on_chain_start(serialized, scrub(inputs), **kwargs)

            def on_chain_end(self, outputs, **kwargs):
                return super().on_chain_end(scrub(outputs), **kwargs)

            def on_chain_error(self, error, **kwargs):
                return super().on_chain_error(RuntimeError(type(error).__name__), **kwargs)

            def on_llm_error(self, error, **kwargs):
                return super().on_llm_error(RuntimeError(type(error).__name__), **kwargs)

        metadata = {
            **trace_metadata(self.settings),
            "message_id": message_id,
            "architecture": self.settings.architecture,
        }
        with propagate_attributes(
            session_id=thread_id, tags=["magenta-rag"], trace_name="Magenta Moments RAG"
        ):
            with self.client.start_as_current_observation(
                name="rag-turn",
                as_type="chain",
                input={
                    "question": question,
                    "history": history_view(history),
                },
                metadata=metadata,
            ) as root:
                reference = {"trace_id": root.trace_id}
                try:
                    reference["trace_url"] = self.client.get_trace_url(trace_id=root.trace_id)
                except Exception:
                    reference["trace_warning"] = WARNING
                trace = SimpleNamespace(
                    reference=reference,
                    callbacks=[BoundedCallback(public_key=self.public_key)],
                    root=root,
                )
                try:
                    yield trace
                except BaseException as error:
                    root.update(
                        level="ERROR",
                        status_message=type(error).__name__,
                        output={"status": "incomplete"},
                    )
                    raise

    def finish(self, trace, result):
        summary = result["messages"][-1].additional_kwargs["result"]
        trace.root.update(
            output={
                "answer": result["messages"][-1].content,
                "answer_status": summary["answer_status"],
                "search_query": summary["search_query"],
                "cited_sources": summary["source_map"],
                "documents": scrub(result.get("documents", [])),
                "observation": summary["observation"],
            }
        )

    def flush(self):
        try:
            self.client.flush()
            if self.exporter:
                failed = self.exporter.failures > self._seen_failures
                self._seen_failures = self.exporter.failures
                if failed:
                    return WARNING
        except Exception:
            return WARNING
        return None


def get_telemetry(settings):
    return Telemetry(settings)
