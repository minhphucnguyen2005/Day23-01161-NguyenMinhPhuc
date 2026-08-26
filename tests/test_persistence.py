"""Tests for checkpointer persistence and state recovery."""

from pathlib import Path

import pytest

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import Route, Scenario, initial_state


def test_build_checkpointer_none() -> None:
    assert build_checkpointer("none") is None


def test_build_checkpointer_memory() -> None:
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = build_checkpointer("memory")
    assert isinstance(checkpointer, MemorySaver)


def test_build_checkpointer_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown checkpointer kind"):
        build_checkpointer("unknown_kind")


def test_build_checkpointer_sqlite_persistence() -> None:
    """Verify SQLite checkpointer persists state and supports state resumption."""
    db_path = "outputs/test_checkpoints.db"
    Path("outputs").mkdir(parents=True, exist_ok=True)
    checkpointer = build_checkpointer("sqlite", database_url=db_path)
    assert checkpointer is not None

    try:
        graph = build_graph(checkpointer=checkpointer)
        scenario = Scenario(
            id="persist-01",
            query="How do I reset my password?",
            expected_route=Route.SIMPLE,
        )
        state = initial_state(scenario)
        config = {"configurable": {"thread_id": "thread-persist-01"}}

        # Execute graph
        result = graph.invoke(state, config=config)
        assert result["route"] == "simple"
        assert result.get("final_answer") is not None

        # Verify state exists in checkpointer
        saved_state = graph.get_state(config)
        assert saved_state.values.get("route") == "simple"
        assert saved_state.values.get("final_answer") == result["final_answer"]

        # Verify state history
        history = list(graph.get_state_history(config))
        assert len(history) > 1, "Graph should record multiple state transitions"
    finally:
        # Close connection cleanly
        if hasattr(checkpointer, "conn") and checkpointer.conn:
            checkpointer.conn.close()
        if Path(db_path).exists():
            try:
                Path(db_path).unlink(missing_ok=True)
            except Exception:
                pass
