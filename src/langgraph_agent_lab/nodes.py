"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, Route, make_event


class ClassificationResult(BaseModel):
    """Structured output schema for intent classification."""

    route: str = Field(
        description="One of: 'risky', 'tool', 'missing_info', 'error', 'simple'"
    )
    risk_level: str = Field(
        default="low",
        description="'high' for risky actions, 'low' otherwise"
    )
    reason: str = Field(
        default="",
        description="Brief reasoning for the classification decision"
    )


# ─── Intake Node ────────────────────────────────────────────────────
def intake_node(state: AgentState) -> dict[str, Any]:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Classification Node ────────────────────────────────────────────
def classify_node(state: AgentState) -> dict[str, Any]:
    """Classify the query into a route using an LLM with structured output."""
    query = state.get("query", "").strip()

    prompt = (
        "You are a customer support triage classifier.\n"
        "Classify the user's support ticket query into exactly one route following this priority:\n"
        "Priority order: risky > tool > missing_info > error > simple\n\n"
        "Route definitions:\n"
        "- 'risky': Actions with financial, destructive, or irreversible side effects "
        "(e.g., refund, delete account, cancel subscription, send external email).\n"
        "- 'tool': Information lookups or database queries "
        "(e.g., check order status, look up user profile, track shipment).\n"
        "- 'missing_info': Vague, ambiguous, or incomplete requests lacking actionable context "
        "(e.g., 'Can you fix it?', 'Help me with this', 'It broke').\n"
        "- 'error': System failures, timeouts, crashes, or unrecoverable error reports "
        "(e.g., 'Timeout failure while processing request', 'System failure cannot recover').\n"
        "- 'simple': General FAQ, how-to questions, or informational guidance without lookups "
        "(e.g., 'How do I reset my password?').\n\n"
        f"User Query: {query}"
    )

    route = Route.SIMPLE.value
    risk_level = "low"

    try:
        llm = get_llm(temperature=0.0)
        if hasattr(llm, "with_structured_output"):
            structured_llm = llm.with_structured_output(ClassificationResult)
            result = structured_llm.invoke(prompt)
        else:
            result = None

        if isinstance(result, ClassificationResult):
            route_val = result.route.strip().lower()
            risk_val = result.risk_level.strip().lower()
        elif isinstance(result, dict):
            route_val = str(result.get("route", "")).strip().lower()
            risk_val = str(result.get("risk_level", "low")).strip().lower()
        else:
            route_val = Route.SIMPLE.value
            risk_val = "low"

        # Validate route value against supported routes
        valid_routes = {
            Route.SIMPLE.value,
            Route.TOOL.value,
            Route.MISSING_INFO.value,
            Route.RISKY.value,
            Route.ERROR.value,
        }
        if route_val in valid_routes:
            route = route_val
        else:
            route = Route.SIMPLE.value

        if route == Route.RISKY.value or risk_val == "high":
            risk_level = "high"
        else:
            risk_level = "low"

    except Exception:
        # Fallback heuristic if LLM is unavailable or offline
        lower_q = query.lower()
        if any(w in lower_q for w in ["refund", "delete", "cancel", "send confirmation"]):
            route = Route.RISKY.value
            risk_level = "high"
        elif any(w in lower_q for w in ["lookup", "status", "order", "track"]):
            route = Route.TOOL.value
            risk_level = "low"
        elif any(w in lower_q for w in ["fix it", "can you fix", "help me"]):
            route = Route.MISSING_INFO.value
            risk_level = "low"
        elif any(w in lower_q for w in ["timeout", "system failure", "crash", "error", "cannot"]):
            route = Route.ERROR.value
            risk_level = "low"
        else:
            route = Route.SIMPLE.value
            risk_level = "low"

    return {
        "route": route,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                f"route={route}",
                route=route,
                risk_level=risk_level,
            )
        ],
    }


# ─── Tool Node ──────────────────────────────────────────────────────
def tool_node(state: AgentState) -> dict[str, Any]:
    """Execute a mock tool call with retry error simulation."""
    route = state.get("route", "")
    attempt = int(state.get("attempt", 0))
    query = state.get("query", "")

    # Simulate transient failures for error-route scenarios
    if route == Route.ERROR.value and attempt < 2:
        result_string = f"ERROR: Transient failure for query '{query}' (attempt {attempt + 1})"
    else:
        result_string = (
            f"SUCCESS: Tool executed successfully for query '{query}'. "
            "Data: Record ID 12345 is Active."
        )

    return {
        "tool_results": [result_string],
        "events": [make_event("tool", "completed", "tool executed", attempt=attempt)],
    }


# ─── Evaluate Node ──────────────────────────────────────────────────
def evaluate_node(state: AgentState) -> dict[str, Any]:
    """Evaluate tool results — the retry-loop gate."""
    tool_results = state.get("tool_results", [])
    latest_result = tool_results[-1] if tool_results else ""

    evaluation_result = "success"

    # LLM-as-judge evaluation if available
    try:
        if "ERROR" in latest_result.upper():
            evaluation_result = "needs_retry"
        else:
            llm = get_llm(temperature=0.0)
            eval_prompt = (
                "You are an evaluator assessing the output of an internal tool execution.\n"
                f"Tool Output: {latest_result}\n\n"
                "Is the tool output satisfactory and successful, or does it need retry?\n"
                "Respond with either 'SUCCESS' or 'NEEDS_RETRY'."
            )
            response = llm.invoke(eval_prompt)
            content = str(getattr(response, "content", response)).strip().upper()
            if "NEEDS_RETRY" in content or "ERROR" in content:
                evaluation_result = "needs_retry"
            else:
                evaluation_result = "success"
    except Exception:
        # Heuristic fallback
        if "ERROR" in latest_result.upper():
            evaluation_result = "needs_retry"
        else:
            evaluation_result = "success"

    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate",
                "completed",
                f"evaluation={evaluation_result}",
                result=evaluation_result,
            )
        ],
    }


# ─── Answer Node ────────────────────────────────────────────────────
def answer_node(state: AgentState) -> dict[str, Any]:
    """Generate a final response using an LLM grounded in context."""
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    route = state.get("route", "")

    context_parts = []
    if tool_results:
        context_parts.append("Tool Results:\n" + "\n".join(tool_results))
    if approval:
        context_parts.append(f"Approval Decision: {approval}")

    context_str = "\n\n".join(context_parts) if context_parts else "No additional tool context."

    prompt = (
        "You are a helpful and professional customer support agent.\n"
        "Answer the user's query clearly, politely, and accurately, grounded in context.\n\n"
        f"Context:\n{context_str}\n\n"
        f"User Query: {query}\n\n"
        "Your Response:"
    )

    try:
        llm = get_llm(temperature=0.3)
        response = llm.invoke(prompt)
        final_answer = str(getattr(response, "content", response)).strip()
    except Exception:
        if route == Route.SIMPLE.value:
            final_answer = (
                f"To resolve your inquiry ('{query}'), "
                "please follow standard procedure in account settings."
            )
        elif tool_results:
            final_answer = f"Your request ('{query}') was processed: {tool_results[-1]}"
        else:
            final_answer = f"Your request ('{query}') has been successfully processed."

    return {
        "final_answer": final_answer,
        "events": [make_event("answer", "completed", "final answer generated")],
    }


# ─── Clarification Node ─────────────────────────────────────────────
def ask_clarification_node(state: AgentState) -> dict[str, Any]:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")

    prompt = (
        "You are a customer support agent. The user's query lacks details.\n"
        f"User Query: '{query}'\n\n"
        "Generate a polite clarification question asking for specific details."
    )

    try:
        llm = get_llm(temperature=0.2)
        response = llm.invoke(prompt)
        question = str(getattr(response, "content", response)).strip()
    except Exception:
        question = f"Could you provide more details regarding '{query}' so we can assist?"

    event = make_event(
        "clarify",
        "completed",
        "clarification requested",
        question=question,
    )
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [event],
    }


# ─── Risky Action Node ──────────────────────────────────────────────
def risky_action_node(state: AgentState) -> dict[str, Any]:
    """Prepare a risky action for human approval."""
    query = state.get("query", "")
    proposed_action = (
        f"Proposed Action: Execute sensitive operations for '{query}'. "
        "Awaiting supervisor authorization."
    )

    return {
        "proposed_action": proposed_action,
        "events": [
            make_event(
                "risky_action",
                "completed",
                "action prepared for approval",
                proposed_action=proposed_action,
            )
        ],
    }


# ─── Approval Node ──────────────────────────────────────────────────
def approval_node(state: AgentState) -> dict[str, Any]:
    """Human-in-the-loop approval step."""
    if os.getenv("LANGGRAPH_INTERRUPT", "false").lower() == "true":
        from langgraph.types import interrupt

        decision = interrupt(
            {"action": state.get("proposed_action"), "query": state.get("query")}
        )
        if isinstance(decision, dict):
            approved = decision.get("approved", True)
            reviewer = decision.get("reviewer", "human-reviewer")
            comment = decision.get("comment", "Interactive HITL approval")
        else:
            approved = bool(decision)
            reviewer = "human-reviewer"
            comment = "Interactive HITL decision"
    else:
        approved = True
        reviewer = "mock-reviewer"
        comment = "Automated policy approval"

    approval_payload = {
        "approved": approved,
        "reviewer": reviewer,
        "comment": comment,
    }

    return {
        "approval": approval_payload,
        "events": [
            make_event(
                "approval",
                "completed",
                f"approval={approved}",
                approved=approved,
            )
        ],
    }


# ─── Retry Node ─────────────────────────────────────────────────────
def retry_or_fallback_node(state: AgentState) -> dict[str, Any]:
    """Record a retry attempt."""
    attempt = int(state.get("attempt", 0)) + 1
    err_msg = f"Attempt {attempt} failed: retrying operation."

    return {
        "attempt": attempt,
        "errors": [err_msg],
        "events": [make_event("retry", "completed", f"retry attempt {attempt}", attempt=attempt)],
    }


# ─── Dead Letter Node ───────────────────────────────────────────────
def dead_letter_node(state: AgentState) -> dict[str, Any]:
    """Handle unresolvable failures after max retries exceeded."""
    query = state.get("query", "")
    attempt = state.get("attempt", 0)
    final_answer = (
        f"Unable to complete the request ('{query}') after {attempt} attempts. "
        "The ticket has been escalated to Tier-3 support (Dead-Letter Queue)."
    )

    return {
        "final_answer": final_answer,
        "events": [make_event("dead_letter", "completed", "escalated to dead letter queue")],
    }


# ─── Finalize Node ──────────────────────────────────────────────────
def finalize_node(state: AgentState) -> dict[str, Any]:
    """Emit a final audit event. All routes must pass through here before END."""
    return {
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
