# Деплой на Streamlit Cloud

## Шаги

1. Зайдите на [share.streamlit.io](https://share.streamlit.io)
2. **Sign in with GitHub**
3. Нажмите **New app → Deploy a public app from GitHub**
4. Заполните поля:
   - **Repository:** `<username>/logistics-rag-assistant`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Откройте **Advanced settings → Secrets** и вставьте:

```toml
GROQ_API_KEY = "ваш_ключ_сюда"
```

6. Нажмите **Deploy**

## Что происходит при первом запуске

Streamlit Cloud не хранит `chroma_db/` — индекс пересобирается автоматически функцией `ensure_index()` в `app.py`. Это занимает 1–2 минуты. При последующих запусках индекс уже есть, старт мгновенный.

## Переменные окружения

| Переменная | Где взять |
|---|---|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) |
