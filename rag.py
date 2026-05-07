"""RAG-конвейер: retrieval + генерация ответа через GigaChat.

Использование:
    from rag import answer
    result = answer("Что делать при повреждении упаковки?")
    print(result["answer"])
    for src in result["sources"]:
        print(src)
"""
from __future__ import annotations

# --- sqlite fix для Streamlit Cloud ---
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
from functools import lru_cache
from typing import TypedDict

import chromadb
from sentence_transformers import SentenceTransformer

import config


# ---------- Типы ----------
class RetrievedChunk(TypedDict):
    text: str
    source: str
    doc_id: str
    score: float


class RagResult(TypedDict):
    answer: str
    sources: list[RetrievedChunk]
    used_llm: bool


# ---------- Кэшируемые ресурсы ----------
@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Загружает модель эмбеддингов один раз."""
    return SentenceTransformer(config.EMBED_MODEL)


@lru_cache(maxsize=1)
def get_collection():
    """Подключается к существующей коллекции в ChromaDB."""
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    return client.get_collection(config.COLLECTION_NAME)


# ---------- Retrieval ----------
def retrieve(query: str, k: int = config.TOP_K) -> list[RetrievedChunk]:
    """Возвращает top-k наиболее релевантных чанков для запроса."""
    embedder = get_embedder()
    query_emb = embedder.encode(
        [config.QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    collection = get_collection()
    results = collection.query(
        query_embeddings=query_emb,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "doc_id": meta["doc_id"],
            "score": 1.0 - dist,  # cosine: distance = 1 - similarity
        })
    return chunks


# ---------- Генерация ----------
def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Форматирует чанки для подачи в промпт."""
    parts = []
    for i, ch in enumerate(chunks, 1):
        parts.append(f"[Источник {i}: {ch['source']}]\n{ch['text']}")
    return "\n\n---\n\n".join(parts)


def _generate_with_llm(question: str, context: str) -> str:
    """Вызывает Groq API через openai-совместимый клиент. Требует GROQ_API_KEY."""
    from openai import OpenAI

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Переменная окружения GROQ_API_KEY не задана. "
            "Получите ключ на https://console.groq.com/keys и добавьте его в .env"
        )

    client = OpenAI(base_url=config.LLM_BASE_URL, api_key=api_key)
    user_prompt = config.USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content


def _stub_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """Заглушка на случай отсутствия ключа API. Просто склеивает релевантные чанки."""
    header = (
        "⚠️ LLM не подключена (не задан GROQ_API_KEY). "
        "Ниже — найденные релевантные фрагменты регламентов:\n\n"
    )
    body = "\n\n---\n\n".join(
        f"📄 {ch['source']}\n{ch['text']}" for ch in chunks
    )
    return header + body


# ---------- Основная функция ----------
def answer(question: str, k: int = config.TOP_K) -> RagResult:
    """Полный пайплайн: retrieval → генерация."""
    chunks = retrieve(question, k=k)
    context = _format_context(chunks)

    used_llm = bool(os.getenv("GROQ_API_KEY"))
    if used_llm:
        try:
            answer_text = _generate_with_llm(question, context)
        except Exception as e:
            answer_text = f"❌ Ошибка вызова Groq: {e}\n\n" + _stub_answer(question, chunks)
            used_llm = False
    else:
        answer_text = _stub_answer(question, chunks)

    return {
        "answer": answer_text,
        "sources": chunks,
        "used_llm": used_llm,
    }


# ---------- CLI для быстрой проверки ----------
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "Что делать, если у водителя нет доверенности?"
    print(f"❓ Вопрос: {q}\n")
    result = answer(q)
    print(f"💬 Ответ:\n{result['answer']}\n")
    print(f"\n📚 Источники (top-{len(result['sources'])}):")
    for i, src in enumerate(result["sources"], 1):
        print(f"  {i}. {src['source']} (score={src['score']:.3f})")
