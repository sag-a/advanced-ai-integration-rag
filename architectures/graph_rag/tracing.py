"""Optional nested stages; telemetry failures never replace application results."""

from contextlib import contextmanager

from rag_core.observability import get_telemetry, scrub
from rag_core.observability import optional_trace


class Stage:
    def __init__(self, settings, name, inputs, output):
        self.settings, self.name, self.inputs, self.output = (
            settings,
            name,
            inputs,
            output,
        )

    @contextmanager
    def turn(self):
        telemetry = get_telemetry(self.settings)
        with telemetry.client.start_as_current_observation(
            name=self.name,
            as_type="span",
            input=scrub(self.inputs),
            metadata={"architecture": "graph_rag"},
        ) as span:
            try:
                yield span
            except BaseException as error:
                span.update(level="ERROR", status_message=type(error).__name__)
                raise
            else:
                span.update(output=scrub(self.output))


@contextmanager
def stage(settings, name, inputs, warnings):
    output = {}
    with optional_trace(
        Stage(settings, name, inputs, output) if settings.langfuse else None
    ) as state:
        yield output
    if state.warning:
        warnings.append(state.warning)
