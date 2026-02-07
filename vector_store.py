"""
Векторное хранилище на базе Weaviate (VDS).
Загрузка документов, умное разбиение на чанки, эмбеддинги через OpenAI, поиск.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from openai import OpenAI
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.data import DataObject

# Загрузка .env из корня проекта
_root = Path(__file__).resolve().parent
env_path = _root / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()

# Размерность вектора OpenAI text-embedding-3-small
OPENAI_EMBEDDING_DIM = 1536
EMBEDDING_MODEL = "text-embedding-3-small"


class VectorStore:
    """Векторное хранилище Weaviate с эмбеддингами OpenAI."""

    def __init__(
        self,
        collection_name: str = "RAGChunk",
        weaviate_http_url: str = None,
        weaviate_grpc_host: str = None,
        weaviate_grpc_port: int = None,
        weaviate_api_key: str = None,
    ):
        """
        Инициализация подключения к Weaviate на VDS.

        Args:
            collection_name: имя коллекции в Weaviate
            weaviate_http_url: URL HTTP API (например http://IP:8080)
            weaviate_grpc_host: хост для gRPC (обычно тот же IP)
            weaviate_grpc_port: порт gRPC (обычно 50051)
            weaviate_api_key: опциональный API-ключ Weaviate
        """
        self.collection_name = collection_name
        http_url = weaviate_http_url or os.getenv("WEAVIATE_HTTP_URL")
        grpc_host = weaviate_grpc_host or os.getenv("WEAVIATE_GRPC_HOST")
        grpc_port = weaviate_grpc_port or int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
        if not http_url:
            raise ValueError("WEAVIATE_HTTP_URL не задан (в .env или аргументом)")

        parsed = urlparse(http_url)
        http_host = parsed.hostname or parsed.path.split(":")[0]
        http_port = parsed.port or 8080
        use_ssl = parsed.scheme == "https"

        if not grpc_host:
            grpc_host = http_host

        auth = None
        if weaviate_api_key or os.getenv("WEAVIATE_API_KEY"):
            from weaviate.classes.init import Auth
            auth = Auth.api_key(weaviate_api_key or os.getenv("WEAVIATE_API_KEY"))

        self._client = weaviate.connect_to_custom(
            http_host=http_host,
            http_port=http_port,
            http_secure=use_ssl,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            grpc_secure=use_ssl,
            auth_credentials=auth,
        )

        if not self._client.is_ready():
            raise RuntimeError("Weaviate не отвечает. Проверьте WEAVIATE_HTTP_URL и доступность портов 8080 и 50051.")

        self._openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        try:
            self._ensure_collection()
        except Exception:
            self._client.close()
            raise
        print(f"Векторное хранилище Weaviate готово. Коллекция: '{collection_name}'")

    def _ensure_collection(self) -> None:
        """Создаёт коллекцию с self-provided векторами, если её ещё нет."""
        if self._client.collections.exists(self.collection_name):
            self._collection = self._client.collections.get(self.collection_name)
            count = self._collection.aggregate.over_all(total_count=True).total_count
            print(f"Коллекция '{self.collection_name}' загружена. Документов: {count}")
            return

        self._client.collections.create(
            name=self.collection_name,
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
            ],
            vector_config=Configure.Vectors.self_provided(),
        )
        self._collection = self._client.collections.get(self.collection_name)
        print(f"Создана коллекция '{self.collection_name}' (векторы передаются при вставке)")

    def _create_embedding(self, text: str) -> List[float]:
        """Эмбеддинг текста через OpenAI."""
        r = self._openai.embeddings.create(input=text, model=EMBEDDING_MODEL)
        return r.data[0].embedding

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Умное разбиение: приоритет абзацам, затем по предложениям, overlap ~20%.
        """
        text = re.sub(r"\s+", " ", text).strip()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= chunk_size:
                current = f"{current}\n\n{para}".strip() if current else para
            elif current:
                chunks.append(current)
                overlap_text = self._get_overlap_text(current, overlap)
                current = f"{overlap_text}\n\n{para}".strip() if overlap_text else para
            else:
                if len(para) > chunk_size:
                    for c in self._split_by_sentences(para, chunk_size, overlap):
                        chunks.append(c)
                    current = ""
                else:
                    current = para

        if current:
            chunks.append(current)
        return [c for c in chunks if len(c) >= 50]

    def _get_overlap_text(self, text: str, size: int) -> str:
        if len(text) <= size:
            return text
        tail = text[-size:]
        for sep in [". ", "! ", "? ", "\n"]:
            i = tail.find(sep)
            if i != -1:
                return tail[i + len(sep) :].strip()
        return tail.strip()

    def _split_by_sentences(self, paragraph: str, chunk_size: int, overlap: int) -> List[str]:
        parts = re.split(r"([.!?]+\s+)", paragraph)
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                sentences.append(parts[i] + parts[i + 1])
            else:
                sentences.append(parts[i])
        if len(parts) % 2 == 1:
            sentences.append(parts[-1])

        result = []
        current = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if len(current) + len(s) + 1 <= chunk_size:
                current = f"{current} {s}".strip() if current else s
            else:
                if current:
                    result.append(current)
                    current = self._get_overlap_text(current, overlap) + " " + s
                else:
                    current = s
        if current:
            result.append(current)
        return result

    def load_documents(self, file_path: str, source: str = None) -> None:
        """
        Загружает текстовый файл: чанки + эмбеддинги в Weaviate.
        Если в коллекции уже есть объекты, загрузка пропускается.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        count = self._collection.aggregate.over_all(total_count=True).total_count
        if count > 0:
            print("В коллекции уже есть документы, загрузка пропущена.")
            return

        chunks = self._chunk_text(text)
        print(f"Текст разбит на {len(chunks)} чанков.")

        source_label = source or path.name
        batch_size = 50
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            objects = []
            for i, chunk in enumerate(batch_chunks):
                emb = self._create_embedding(chunk)
                objects.append(
                    DataObject(
                        properties={"content": chunk, "source": source_label},
                        vector=emb,
                    )
                )
                if (start + i + 1) % 10 == 0:
                    print(f"Обработано {start + i + 1}/{len(chunks)} чанков.")
            self._collection.data.insert_many(objects)
        print(f"Загружено {len(chunks)} чанков в Weaviate (коллекция '{self.collection_name}').")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Поиск релевантных чанков по запросу."""
        query_vector = self._create_embedding(query)
        result = self._collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            return_metadata=["distance"],
        )

        docs = []
        for obj in result.objects:
            props = obj.properties or {}
            docs.append({
                "id": str(obj.uuid),
                "text": props.get("content", ""),
                "source": props.get("source", ""),
                "distance": obj.metadata.distance if obj.metadata else None,
            })
        return docs

    def get_collection_stats(self) -> Dict[str, Any]:
        """Статистика коллекции."""
        try:
            total = self._collection.aggregate.over_all(total_count=True).total_count
        except Exception:
            total = 0
        return {
            "name": self.collection_name,
            "count": total,
            "backend": "Weaviate",
        }

    def close(self) -> None:
        """Закрыть соединение с Weaviate."""
        if hasattr(self, "_client") and self._client:
            self._client.close()


if __name__ == "__main__":
    import sys
    if not os.getenv("OPENAI_API_KEY"):
        print("Установите OPENAI_API_KEY")
        sys.exit(1)
    if not os.getenv("WEAVIATE_HTTP_URL"):
        print("Установите WEAVIATE_HTTP_URL (например http://IP:8080)")
        sys.exit(1)

    store = VectorStore(collection_name="RAGChunk_test")
    try:
        stats = store.get_collection_stats()
        print("Статистика:", stats)
        if stats["count"] == 0 and Path("data/docs.txt").exists():
            store.load_documents("data/docs.txt")
        results = store.search("Что такое RAG?", top_k=2)
        print("Поиск 'Что такое RAG?':", [r["text"][:80] + "..." for r in results])
    finally:
        store.close()
