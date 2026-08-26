"""Routing functions for conditional edges.

Each function takes AgentState and returns a string — the name of the next node.
These strings MUST match node names registered in graph.py.
"""

from __future__ import annotations

from typing import Any

from .state import AgentState, Route


def route_after_classify(state: AgentState | dict[str, Any]) -> str:
    """Map classified route to the next graph node."""
    route = str(state.get("route", "")).lower()
    mapping = {
        Route.SIMPLE.value: "answer",
        Route.TOOL.value: "tool",
        Route.MISSING_INFO.value: "clarify",
        Route.RISKY.value: "risky_action",
        Route.ERROR.value: "retry",
    }
    return mapping.get(route, "answer")


def route_after_evaluate(state: AgentState | dict[str, Any]) -> str:
    """Decide if tool result is satisfactory or needs retry."""
    eval_res = state.get("evaluation_result", "success")
    if eval_res == "needs_retry":
        return "retry"
    return "answer"


def route_after_retry(state: AgentState | dict[str, Any]) -> str:
    """Decide whether to retry the tool or give up."""
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 3))
    if attempt < max_attempts:
        return "tool"
    return "dead_letter"


def route_after_approval(state: AgentState | dict[str, Any]) -> str:
    """Route based on human approval decision."""
    approval = state.get("approval")
    if isinstance(approval, dict):
        is_approved = bool(approval.get("approved", False))
    elif approval is not None and hasattr(approval, "approved"):
        is_approved = bool(approval.approved)
    else:
        is_approved = bool(approval)

    if is_approved:
        return "tool"
    return "clarify"
