"""
Кеш ответов RAG на SQLite: вопрос → ответ + контекст, быстрый повтор без поиска и LLM.
"""

import hashlib
import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional


class RAGCache:
    """Кеш пар вопрос–ответ для RAG (SQLite)."""

    def __init__(self, db_path: str = "rag_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    context TEXT,
                    created_at TEXT
                )
            """)

    def _hash(self, query: str) -> str:
        normalized = " ".join(query.lower().strip().split())
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Получить ответ из кеша по вопросу."""
        h = self._hash(query)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT query, answer, context, created_at FROM cache WHERE query_hash = ?",
                (h,),
            ).fetchone()
        if not row:
            return None
        return {
            "query": row[0],
            "answer": row[1],
            "context": json.loads(row[2]) if row[2] else None,
            "created_at": row[3],
        }

    def set(self, query: str, answer: str, context: Optional[List[str]] = None) -> None:
        """Сохранить ответ в кеш."""
        h = self._hash(query)
        ctx_json = json.dumps(context) if context else None
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (query_hash, query, answer, context, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (h, query, answer, ctx_json, now),
            )

    def clear(self) -> None:
        """Очистить весь кеш."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")

    def get_stats(self) -> Dict[str, Any]:
        """Статистика кеша."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            row = conn.execute(
                "SELECT MIN(created_at), MAX(created_at) FROM cache"
            ).fetchone()
        size_mb = 0.0
        if os.path.exists(self.db_path):
            size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
        return {
            "total_entries": total,
            "oldest_entry": row[0],
            "newest_entry": row[1],
            "db_size_mb": round(size_mb, 2),
        }
