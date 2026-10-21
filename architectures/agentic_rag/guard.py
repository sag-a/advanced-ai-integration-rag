"""Code-enforced limits shared by the coordinator and its research helper."""

from pathlib import PurePosixPath
from langchain_core.messages import ToolMessage
from langchain.agents.middleware import AgentMiddleware
from rag_core.config import AppError
from .tools import ACTOR

TOOLS = {
    "search_pdf",
    "read_clause",
    "assess_evidence",
    "select_evidence",
    "read_file",
    "ls",
    "glob",
    "grep",
    "write_file",
    "edit_file",
    "write_todos",
    "task",
}


class ResearchGuard(AgentMiddleware):
    def __init__(self, research, *, helper=False):
        self.research, self.helper = research, helper
        self.loaded_skills = set()

    def allowed(self, name):
        return name in TOOLS and not (self.helper and name in {"task", "select_evidence"})

    def wrap_model_call(self, request, handler):
        r = self.research
        with r.lock:
            r.check()
            if r.model_calls >= r.settings.max_model_calls:
                raise AppError("Agent model-call budget exhausted; no answer was accepted.")
            r.model_calls += 1
        tools = [
            t
            for t in request.tools
            if self.allowed(t.name if hasattr(t, "name") else t.get("name", ""))
        ]
        response = handler(request.override(tools=tools))
        with r.lock:
            r.check()
            for message in response.result:
                if getattr(message, "usage_metadata", None):
                    r.usage.append(message.usage_metadata)
        return response

    def wrap_tool_call(self, request, handler):
        r = self.research
        name, args = request.tool_call["name"], request.tool_call["args"]
        delegation = name == "task"
        with r.lock:
            r.check()
            if not self.allowed(name):
                raise AppError("Tool permission denied in the PDF research lesson.")
            if r.tool_calls >= r.settings.max_tool_calls:
                raise AppError("Global research tool-call budget exhausted.")
            r.tool_calls += 1
            if name in {"search_pdf", "read_clause"} and not self.loaded_skills:
                r.event(
                    "skill_required",
                    actor="researcher" if self.helper else "coordinator",
                )
                return ToolMessage(
                    content="Read the matching /skills/membership/SKILL.md, /skills/hearts/SKILL.md or /skills/coupons/SKILL.md before researching this topic.",
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
            if name in {"read_file", "write_file", "edit_file"}:
                path = PurePosixPath(args.get("file_path", ""))
                if (
                    ".." in path.parts
                    or not path.is_absolute()
                    or not str(path).startswith(("/skills/", "/workspace/"))
                ):
                    raise AppError("Only virtual /skills/ and /workspace/ paths are allowed.")
                if name != "read_file" and not str(path).startswith("/workspace/"):
                    raise AppError("Skill files are read-only.")
                if name == "read_file" and str(path).startswith("/skills/"):
                    if str(path) not in {
                        f"/skills/{name}/SKILL.md"
                        for name in ("membership", "hearts", "coupons")
                    }:
                        raise AppError("Select one of the bundled research skills.")
            if delegation:
                if (
                    args.get("subagent_type") != "general-purpose"
                    or r.delegations >= r.settings.max_delegations
                    or r.active_delegations
                ):
                    raise AppError(
                        "Only one PDF researcher at a time is allowed, within the delegation budget."
                    )
                r.delegations += 1
                r.active_delegations += 1
                r.event("delegate", researcher="PDF researcher")
        actor_token = ACTOR.set("researcher" if self.helper else "coordinator")
        try:
            result = handler(request)
            if (
                name == "read_file"
                and args.get("file_path", "").startswith("/skills/")
                and getattr(result, "status", None) != "error"
            ):
                self.loaded_skills.add(args["file_path"])
                r.event("skill_read", path=args["file_path"])
            r.check()
            return result
        except Exception as error:
            r.failure = (
                f"Research tool failed ({type(error).__name__}); no answer was accepted."
            )
            raise
        finally:
            ACTOR.reset(actor_token)
            if delegation:
                with r.lock:
                    r.active_delegations -= 1
