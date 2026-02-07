"""
RAG-пайплайн: кеш → Weaviate → OpenAI.
Запрос → (кеш?) → поиск чанков → промпт с контекстом → ответ LLM → сохранение в кеш.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from cache import RAGCache
from vector_store import VectorStore

_root = Path(__file__).resolve().parent
if (_root / ".env").exists():
    load_dotenv(_root / ".env")
else:
    load_dotenv()


class RAGPipeline:
    """Единый пайплайн: кеш (SQLite) + векторное хранилище (Weaviate) + LLM (OpenAI)."""

    def __init__(
        self,
        collection_name: str = "RAGChunk",
        cache_db_path: str = "rag_cache.db",
        data_file: str = "data/docs.txt",
        model: str = "gpt-4o-mini",
    ):
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY не задан")
        if not os.getenv("WEAVIATE_HTTP_URL"):
            raise ValueError("WEAVIATE_HTTP_URL не задан")

        self.model = model
        self._openai = OpenAI()

        print("Инициализация векторного хранилища (Weaviate)...")
        self.vector_store = VectorStore(collection_name=collection_name)

        data_path = _root / data_file
        count = self.vector_store.get_collection_stats().get("count", 0)
        if count == 0 and data_path.exists():
            print(f"Загрузка документов из {data_file}...")
            self.vector_store.load_documents(str(data_path))

        print("Инициализация кеша...")
        self.cache = RAGCache(db_path=cache_db_path)
        print("RAG Pipeline готов (OpenAI + Weaviate).")

    def _build_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        context_parts = [
            f"Документ {i}:\n{doc.get('text', doc.get('content', ''))}\n"
            for i, doc in enumerate(context_docs, 1)
        ]
        context = "\n".join(context_parts)
        return f"""Ты — полезный AI-ассистент. Ответь на вопрос только на основе контекста ниже.

Контекст:
{context}

Вопрос: {query}

Правила: отвечай кратко и по делу; если в контексте нет ответа — так и скажи; отвечай на русском."""

    def _generate(self, prompt: str) -> str:
        resp = self._openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Ты отвечаешь на вопросы строго по предоставленному контексту."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        return (resp.choices[0].message.content or "").strip()

    def query(self, user_query: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Обработка запроса: кеш → поиск в Weaviate → генерация ответа → кеш.
        """
        if use_cache:
            cached = self.cache.get(user_query)
            if cached:
                return {
                    "query": user_query,
                    "answer": cached["answer"],
                    "from_cache": True,
                    "context_docs": [],
                    "cached_at": cached.get("created_at"),
                }

        context_docs = self.vector_store.search(user_query, top_k=5)
        prompt = self._build_prompt(user_query, context_docs)
        answer = self._generate(prompt)

        if use_cache:
            ctx_for_cache = [d.get("text", d.get("content", "")) for d in context_docs]
            self.cache.set(user_query, answer, ctx_for_cache)

        return {
            "query": user_query,
            "answer": answer,
            "from_cache": False,
            "context_docs": context_docs,
            "model": self.model,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_store": self.vector_store.get_collection_stats(),
            "cache": self.cache.get_stats(),
            "model": self.model,
        }


if __name__ == "__main__":
    import sys
    try:
        pipeline = RAGPipeline()
        r = pipeline.query("Что такое RAG?")
        print("Ответ:", r["answer"])
        print("Из кеша:", r["from_cache"])
        print("Статистика:", pipeline.get_stats())
    except Exception as e:
        print("Ошибка:", e)
        sys.exit(1)
