"""
Консольный RAG-ассистент: OpenAI + Weaviate (VDS).
Команды: exit/quit — выход, stats — статистика, clear — очистка кеша.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from rag_pipeline import RAGPipeline

_root = Path(__file__).resolve().parent
if (_root / ".env").exists():
    load_dotenv(_root / ".env")
else:
    load_dotenv()


def main() -> None:
    print("=" * 60)
    print("  RAG Ассистент (OpenAI + Weaviate на VDS)")
    print("=" * 60)
    print("Команды: exit / quit — выход, stats — статистика, clear — очистка кеша\n")

    if not os.getenv("OPENAI_API_KEY"):
        print("Ошибка: задайте OPENAI_API_KEY в .env или окружении.")
        sys.exit(1)
    if not os.getenv("WEAVIATE_HTTP_URL"):
        print("Ошибка: задайте WEAVIATE_HTTP_URL в .env (например http://IP:8080).")
        sys.exit(1)

    try:
        pipeline = RAGPipeline(
            collection_name="RAGChunk",
            cache_db_path="rag_cache.db",
            data_file="data/docs.txt",
            model="gpt-4o-mini",
        )
    except Exception as e:
        print(f"Ошибка инициализации: {e}")
        sys.exit(1)

    print("\nГотов к вопросам.\n")

    try:
        while True:
            try:
                user_input = input("Вопрос: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nВыход.")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("До свидания.")
                break
            if user_input.lower() == "stats":
                stats = pipeline.get_stats()
                print("\nСтатистика:")
                print("  Weaviate:", stats["vector_store"])
                print("  Кеш:", stats["cache"])
                print("  Модель:", stats["model"])
                print()
                continue
            if user_input.lower() == "clear":
                pipeline.cache.clear()
                print("Кеш очищен.\n")
                continue

            result = pipeline.query(user_input)
            print()
            if result.get("from_cache"):
                print("[из кеша]")
            print("Ответ:", result["answer"])
            if result.get("context_docs"):
                print("  (использовано чанков:", len(result["context_docs"]), ")")
            print()
    finally:
        if hasattr(pipeline, "vector_store") and hasattr(pipeline.vector_store, "close"):
            pipeline.vector_store.close()


if __name__ == "__main__":
    main()
