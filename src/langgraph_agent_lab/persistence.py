"""Checkpointer adapter."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def build_checkpointer(kind: str = "memory", database_url: str | None = None) -> Any | None:  # noqa: ANN401
    """Return a LangGraph checkpointer.

    Supports:
    - 'none': no checkpointer
    - 'memory': in-memory MemorySaver
    - 'sqlite': persistent SQLite SqliteSaver
    - 'postgres': Postgres saver
    """
    if kind == "none" or kind is None:
        return None

    if kind == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    if kind == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "langgraph-checkpoint-sqlite is required for sqlite checkpointer. "
                "Install with: pip install langgraph-checkpoint-sqlite"
            ) from exc

        db_path = database_url or "outputs/checkpoints.db"
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL;")
        return SqliteSaver(conn=conn)

    if kind == "postgres":
        if not database_url:
            raise ValueError("database_url is required for postgres checkpointer")
        try:
            from langgraph.checkpoint.postgres import (  # type: ignore[import-not-found,import-untyped]
                PostgresSaver,
            )

            return PostgresSaver.from_conn_string(database_url)
        except ImportError as exc:
            raise RuntimeError("Install: pip install langgraph-checkpoint-postgres") from exc

    raise ValueError(f"Unknown checkpointer kind: {kind}")
