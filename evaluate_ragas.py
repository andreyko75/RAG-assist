"""
Оценка качества RAG через RAGAS (Faithfulness, Context Precision).
Запуск: python evaluate_ragas.py
"""

import os
import sys
import warnings
from pathlib import Path

# Подавляем предупреждения RAGAS по тексту сообщения (без импорта классов)
warnings.filterwarnings("ignore", message=".*deprecated.*")
warnings.filterwarnings("ignore", message=".*unclosed.*")
warnings.filterwarnings("ignore", message=".*tracemalloc.*")

from dotenv import load_dotenv

_root = Path(__file__).resolve().parent
if (_root / ".env").exists():
    load_dotenv(_root / ".env")
else:
    load_dotenv()

from datasets import Dataset
from rag_pipeline import RAGPipeline

# Legacy-метрики (уже экземпляры) — совместимы с evaluate()
try:
    from ragas.metrics._faithfulness import faithfulness
    from ragas.metrics._context_precision import context_precision
    RAGAS_METRICS = [faithfulness, context_precision]
except ImportError:
    try:
        from ragas.metrics import faithfulness, context_precision
        RAGAS_METRICS = [faithfulness, context_precision]
    except ImportError:
        print("Установите ragas: pip install ragas")
        sys.exit(1)

from ragas import evaluate


EVALUATION_QUESTIONS = [
    "Что такое машинное обучение?",
    "Какие основные типы машинного обучения существуют?",
    "Что такое нейронная сеть?",
    "Как работают трансформеры в NLP?",
    "Что такое RAG и как он работает?",
]


def main() -> None:
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("WEAVIATE_HTTP_URL"):
        print("Задайте OPENAI_API_KEY и WEAVIATE_HTTP_URL в .env")
        sys.exit(1)

    print("Инициализация RAG (OpenAI + Weaviate)...")
    pipeline = RAGPipeline(
        collection_name="RAGChunk",
        cache_db_path="rag_cache.db",
        data_file="data/docs.txt",
        model="gpt-4o-mini",
    )

    try:
        _run_evaluation(pipeline)
    finally:
        if hasattr(pipeline, "vector_store") and hasattr(pipeline.vector_store, "close"):
            pipeline.vector_store.close()


def _run_evaluation(pipeline: RAGPipeline) -> None:
    questions_list = []
    answers_list = []
    contexts_list = []
    ground_truth_list = []

    print("Получение ответов на тестовые вопросы...")
    for i, q in enumerate(EVALUATION_QUESTIONS, 1):
        print(f"  {i}/{len(EVALUATION_QUESTIONS)}: {q}")
        r = pipeline.query(q, use_cache=False)
        questions_list.append(q)
        answers_list.append(r["answer"])
        contexts_list.append([d.get("text", d.get("content", "")) for d in r["context_docs"]])
        ground_truth_list.append(r["answer"][:200])  # упрощённый ground truth для демо

    dataset = Dataset.from_dict({
        "question": questions_list,
        "answer": answers_list,
        "contexts": contexts_list,
        "ground_truth": ground_truth_list,
    })

    print("\nЗапуск RAGAS (Faithfulness, Context Precision)...")
    try:
        result = evaluate(
            dataset=dataset,
            metrics=RAGAS_METRICS,
        )
    except Exception as e:
        print(f"Ошибка RAGAS: {e}")
        raise

    # Формируем расширенный отчёт
    _print_report(result)


def _print_report(result) -> None:
    """Расширенный отчёт по результатам RAGAS."""
    sep = "=" * 70
    print(f"\n{sep}")
    print("  ОТЧЁТ ОЦЕНКИ RAG (RAGAS)")
    print(f"{sep}\n")

    if not hasattr(result, "_repr_dict"):
        print("  faithfulness:", result["faithfulness"])
        print("  context_precision:", result["context_precision"])
        print("\nГотово.")
        return

    # Средние по метрикам
    means = result._repr_dict
    print("  СРЕДНИЕ МЕТРИКИ (по всем вопросам)")
    print("  " + "-" * 50)
    for name, value in means.items():
        label = "faithfulness (верность ответа контексту)"
        if "context_precision" in name:
            label = "context_precision (релевантность подобранного контекста)"
        print(f"    {name}: {value:.4f}  — {label}")
    print()

    # По каждому вопросу
    f_scores = result["faithfulness"] if "faithfulness" in result._scores_dict else []
    cp_scores = result["context_precision"] if "context_precision" in result._scores_dict else []
    n = max(len(f_scores), len(cp_scores), len(EVALUATION_QUESTIONS))

    if n > 0:
        print("  ДЕТАЛИЗАЦИЯ ПО ВОПРОСАМ")
        print("  " + "-" * 50)
        for i in range(n):
            q_short = (EVALUATION_QUESTIONS[i][:45] + "…") if len(EVALUATION_QUESTIONS[i]) > 45 else EVALUATION_QUESTIONS[i]
            f_val = f_scores[i] if i < len(f_scores) else "—"
            cp_val = cp_scores[i] if i < len(cp_scores) else "—"
            if isinstance(f_val, float):
                f_str = f"{f_val:.4f}"
            else:
                f_str = str(f_val)
            if isinstance(cp_val, float):
                cp_str = f"{cp_val:.4f}"
            else:
                cp_str = str(cp_val)
            print(f"    {i + 1}. {q_short}")
            print(f"       faithfulness: {f_str}  |  context_precision: {cp_str}")
        print()

    # Итоговая оценка
    avg = sum(means.values()) / len(means) if means else 0
    print("  ИТОГО")
    print("  " + "-" * 50)
    print(f"    Средний балл (по всем метрикам): {avg:.4f}")
    if avg >= 0.85:
        print("    Оценка: отличное качество RAG.")
    elif avg >= 0.7:
        print("    Оценка: хорошее качество, допустимо для продакшена.")
    elif avg >= 0.5:
        print("    Оценка: удовлетворительно, стоит улучшить чанкинг или промпты.")
    else:
        print("    Оценка: требуется доработка данных и/или поиска.")
    print()
    print("  Пояснения:")
    print("    • faithfulness — насколько ответ опирается только на контекст (1.0 = без галлюцинаций).")
    print("    • context_precision — насколько подобранные чанки релевантны вопросу (1.0 = идеальный подбор).")
    print(f"\n{sep}\nГотово.")


if __name__ == "__main__":
    main()
