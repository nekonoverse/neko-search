"""SQLite document store for neko-search."""

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentStore:
    """SQLite-backed document store."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS documents (
                note_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                tokens TEXT NOT NULL,
                published TEXT
            )"""
        )
        self._conn.commit()
        logger.info("DocumentStore opened: %s", db_path)

    @property
    def doc_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return row[0] if row else 0

    def get(self, note_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT note_id, text, tokens, published FROM documents WHERE note_id = ?",
            (note_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "note_id": row[0],
            "text": row[1],
            "tokens": json.loads(row[2]),
            "published": row[3],
        }

    def upsert(self, note_id: str, text: str, tokens: list[str], published: str | None = None):
        self._conn.execute(
            """INSERT INTO documents (note_id, text, tokens, published)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(note_id) DO UPDATE SET
                 text=excluded.text, tokens=excluded.tokens, published=excluded.published""",
            (note_id, text, json.dumps(tokens), published),
        )
        self._conn.commit()

    def bulk_upsert(self, docs: list[dict]):
        """Upsert multiple documents in a single transaction."""
        with self._conn:
            self._conn.executemany(
                """INSERT INTO documents (note_id, text, tokens, published)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(note_id) DO UPDATE SET
                     text=excluded.text, tokens=excluded.tokens, published=excluded.published""",
                [
                    (d["note_id"], d["text"], json.dumps(d["tokens"]), d.get("published"))
                    for d in docs
                ],
            )

    def delete(self, note_id: str) -> bool:
        cursor = self._conn.execute("DELETE FROM documents WHERE note_id = ?", (note_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def iter_all(self, batch_size: int = 1000):
        """Iterate all documents in batches for index rebuilding."""
        cursor = self._conn.execute("SELECT note_id, text, tokens, published FROM documents")
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            for row in rows:
                yield {
                    "note_id": row[0],
                    "text": row[1],
                    "tokens": json.loads(row[2]),
                    "published": row[3],
                }

    def export_texts(self) -> list[str]:
        """Export all document texts for corpus training."""
        cursor = self._conn.execute("SELECT text FROM documents")
        return [row[0] for row in cursor.fetchall()]

    def close(self):
        self._conn.close()
