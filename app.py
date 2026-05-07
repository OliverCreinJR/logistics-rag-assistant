"""Streamlit UI для RAG-ассистента по логистическим процедурам.

Запуск локально:
    streamlit run app.py

Деплой:
    push в GitHub → Streamlit Cloud (https://share.streamlit.io) → New app.
    В настройках app.py добавьте секрет GIGACHAT_CREDENTIALS.
"""
from __future__ import annotations

# sqlite swap — должен быть до любого импорта chromadb (нужен на Streamlit Cloud)
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Подгружаем .env (для локального запуска)
load_dotenv()

# На Streamlit Cloud секреты приходят через st.secrets — пробрасываем их в env,
# чтобы остальной код продолжал работать через os.getenv
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except (FileNotFoundError, Exception):
    # Локально файла secrets.toml нет — это нормально, ключ берётся из .env
    pass

import config


# ---------- Авто-индексация при первом запуске ----------
# Если индекса ещё нет, собираем его на лету (актуально для Streamlit Cloud,
# где chroma_db/ может не быть в репозитории).
def ensure_index() -> None:
    """Проверяет, что коллекция ChromaDB существует и непустая. Если нет — строит индекс."""
    import chromadb
    needs_build = False
    try:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        collection = client.get_collection(config.COLLECTION_NAME)
        if collection.count() == 0:
            needs_build = True
    except Exception:
        # Коллекции нет, или ChromaDB не может её найти — строим с нуля
        needs_build = True

    if needs_build:
        with st.spinner("⚙️ Первый запуск: индексация регламентов (1–2 минуты)..."):
            from ingest import main as build_index
            build_index()


# ---------- UI ----------
st.set_page_config(
    page_title="Логистический ассистент",
    page_icon="📦",
    layout="centered",
)

st.title("📦 Ассистент по логистическим процедурам")
st.caption(
    "RAG-система: семантический поиск по корпоративным регламентам + генерация ответа GigaChat. "
    "Учебный pet-проект."
)

# Подготовка индекса
ensure_index()

# Импорт после индексации, чтобы не упасть на пустой коллекции
from rag import answer  # noqa: E402

# Боковая панель: статус и примеры
with st.sidebar:
    st.subheader("Статус системы")
    has_key = bool(os.getenv("GROQ_API_KEY"))
    if has_key:
        st.success("✅ Groq подключён")
    else:
        st.warning(
            "⚠️ GROQ_API_KEY не задан.\n\n"
            "Без ключа система покажет найденные фрагменты регламентов, но не сгенерирует "
            "связный ответ. Получите ключ на console.groq.com/keys."
        )
    st.divider()
    st.subheader("Примеры вопросов")
    examples = [
        "Что делать, если при приёмке обнаружено повреждение упаковки?",
        "Какие документы нужны перевозчику для въезда на склад?",
        "В какой срок нужно подать претензию перевозчику?",
        "Что делать, если артикул не проходит в WMS?",
        "Как оформить возврат поставщику?",
    ]
    for ex in examples:
        if st.button(ex, key=ex, use_container_width=True):
            st.session_state["question_input"] = ex

    st.divider()
    st.caption(f"📂 Документов в базе: {len(list(config.DOCS_DIR.glob('*.md')))}")
    st.caption(f"🧠 Эмбеддер: `{config.EMBED_MODEL.split('/')[-1]}`")
    st.caption(f"💬 LLM: `{config.LLM_MODEL}`")


# Основное поле ввода
question = st.text_area(
    "Ваш вопрос:",
    value=st.session_state.get("question_input", ""),
    height=100,
    placeholder="Например: что делать, если в ТТН неверный артикул?",
    key="question_text",
)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("🔍 Спросить", type="primary", use_container_width=True)


# Обработка запроса
if submit and question.strip():
    with st.spinner("Ищу в регламентах..."):
        result = answer(question.strip())

    st.divider()

    # Ответ
    st.subheader("💬 Ответ")
    if result["used_llm"]:
        st.markdown(result["answer"])
    else:
        st.info(result["answer"])

    # Источники
    st.divider()
    st.subheader(f"📚 Источники (top-{len(result['sources'])})")
    for i, src in enumerate(result["sources"], 1):
        with st.expander(
            f"#{i} — {src['source']} (релевантность: {src['score']:.2f})",
            expanded=False,
        ):
            st.markdown(src["text"])

elif submit and not question.strip():
    st.warning("Введите вопрос.")


# Подвал
st.divider()
st.caption(
    "Система отвечает только на основе регламентов из базы знаний. "
    "При отсутствии информации сообщает об этом и не выдумывает ответ."
)
