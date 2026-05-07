"""Индексация регламентов в ChromaDB.

Запуск:
    python ingest.py

Что делает:
    1. Читает все .md файлы из docs/
    2. Режет каждый документ на чанки с пересечением
    3. Эмбеддит чанки моделью multilingual-e5
    4. Сохраняет в локальную ChromaDB (chroma_db/)

Запускается один раз при изменении документов. Индекс persistent.
"""
from __future__ import annotations

# --- sqlite fix для Streamlit Cloud (на локальной машине no-op) ---
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import config


def load_documents() -> list[dict]:
    """Читает все markdown-файлы из docs/."""
    documents = []
    for md_path in sorted(config.DOCS_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        documents.append({
            "id": md_path.stem,
            "source": md_path.name,
            "text": text,
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Режет документы на чанки с метаданными."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in documents:
        for i, piece in enumerate(splitter.split_text(doc["text"])):
            chunks.append({
                "id": f"{doc['id']}__{i:03d}",
                "text": piece,
                "metadata": {
                    "source": doc["source"],
                    "doc_id": doc["id"],
                    "chunk_index": i,
                },
            })
    return chunks


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[list[float]]:
    """Эмбеддит чанки. Применяет префикс passage: согласно требованиям e5."""
    texts = [config.PASSAGE_PREFIX + c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,  # для cosine similarity
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def main() -> None:
    print(f"📂 Чтение документов из {config.DOCS_DIR}")
    documents = load_documents()
    print(f"   Загружено документов: {len(documents)}")

    print(f"\n✂️  Чанкование (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    chunks = chunk_documents(documents)
    print(f"   Получено чанков: {len(chunks)}")

    print(f"\n🧠 Загрузка модели эмбеддингов: {config.EMBED_MODEL}")
    model = SentenceTransformer(config.EMBED_MODEL)

    print(f"\n🔢 Вычисление эмбеддингов")
    embeddings = embed_chunks(chunks, model)

    print(f"\n💾 Запись в ChromaDB: {config.CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    # Пересоздаём коллекцию, чтобы при повторных запусках не дублировать данные
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[c["metadata"] for c in chunks],
    )

    print(f"\n✅ Готово. В коллекции '{config.COLLECTION_NAME}': {collection.count()} чанков")


if __name__ == "__main__":
    main()
