"""Graph smoke tests.

These tests verify end-to-end graph execution.
"""

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


def test_graph_runs_and_routes_correctly() -> None:
    """Verify graph executes sample queries across routes."""
    test_cases = [
        ("How do I reset my password?", Route.SIMPLE.value),
        ("Please lookup order status for order 123", Route.TOOL.value),
        ("Refund this customer and cancel order", Route.RISKY.value),
        ("Can you fix it?", Route.MISSING_INFO.value),
        ("Timeout failure while processing request", Route.ERROR.value),
    ]
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    for query, expected_route in test_cases:
        scenario = Scenario(id="smoke", query=query, expected_route=Route(expected_route))
        state = initial_state(scenario)
        result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
        assert result["route"] == expected_route
        assert result.get("final_answer") or result.get("pending_question")


def test_graph_terminates_all_routes() -> None:
    """Verify every route reaches finalize node."""
    graph = build_graph(checkpointer=build_checkpointer("memory"))
    queries = [
        ("simple query about help", Route.SIMPLE),
        ("lookup order status 999", Route.TOOL),
        ("fix it", Route.MISSING_INFO),
        ("delete user account now", Route.RISKY),
        ("timeout error in system", Route.ERROR),
    ]
    for query, route in queries:
        scenario = Scenario(id=f"term-{route.value}", query=query, expected_route=route)
        state = initial_state(scenario)
        result = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
        events = result.get("events", [])
        finalize_events = [e for e in events if e.get("node") == "finalize"]
        assert finalize_events, f"Route {route.value} did not reach finalize node"
