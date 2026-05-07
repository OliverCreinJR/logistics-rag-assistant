"""Оценка качества retrieval.

Запуск:
    python eval.py

Что делает:
    Прогоняет 15 тестовых вопросов через retrieve(),
    проверяет, попал ли ожидаемый документ в top-1, top-3, top-5.
    Печатает hit rate и сохраняет таблицу результатов в eval_results.md.

Эта метрика — ключевой "AI-эксперимент" проекта, цифру кладёте в README и в мотивационное письмо.
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from rag import retrieve


def evaluate() -> None:
    questions = json.loads(config.EVAL_FILE.read_text(encoding="utf-8"))

    results = []
    hits = {1: 0, 3: 0, 5: 0}

    print(f"🧪 Прогон {len(questions)} вопросов\n")
    for i, item in enumerate(questions, 1):
        retrieved = retrieve(item["question"], k=5)
        retrieved_doc_ids = [r["doc_id"] for r in retrieved]
        expected = item["expected_doc"]

        # Проверка попадания в top-1, top-3, top-5
        in_top = {k: expected in retrieved_doc_ids[:k] for k in (1, 3, 5)}
        for k, v in in_top.items():
            if v:
                hits[k] += 1

        # Позиция ожидаемого документа (или -1, если не найден)
        try:
            position = retrieved_doc_ids.index(expected) + 1
        except ValueError:
            position = -1

        status = "✅" if in_top[3] else "❌"
        print(f"{status} [{i:2d}/{len(questions)}] позиция={position:>2}  «{item['question'][:60]}…»")

        results.append({
            "question": item["question"],
            "expected": expected,
            "position": position,
            "top1": in_top[1],
            "top3": in_top[3],
            "top5": in_top[5],
            "retrieved": retrieved_doc_ids,
        })

    # Подсчёт метрик
    n = len(questions)
    metrics = {k: hits[k] / n for k in (1, 3, 5)}

    print("\n" + "=" * 60)
    print(f"📊 Результаты ({n} вопросов)")
    print("=" * 60)
    print(f"  Hit Rate @ 1: {metrics[1]:.1%}  ({hits[1]}/{n})")
    print(f"  Hit Rate @ 3: {metrics[3]:.1%}  ({hits[3]}/{n})")
    print(f"  Hit Rate @ 5: {metrics[5]:.1%}  ({hits[5]}/{n})")

    # Сохранение отчёта
    report_path = Path(__file__).parent / "eval_results.md"
    lines = [
        "# Результаты оценки retrieval",
        "",
        f"**Модель эмбеддингов:** `{config.EMBED_MODEL}`  ",
        f"**Размер чанка:** {config.CHUNK_SIZE} символов, overlap {config.CHUNK_OVERLAP}  ",
        f"**Тестовый набор:** {n} вопросов",
        "",
        "## Сводные метрики",
        "",
        "| Метрика | Значение | Попаданий |",
        "|---|---|---|",
        f"| Hit Rate @ 1 | **{metrics[1]:.1%}** | {hits[1]} / {n} |",
        f"| Hit Rate @ 3 | **{metrics[3]:.1%}** | {hits[3]} / {n} |",
        f"| Hit Rate @ 5 | **{metrics[5]:.1%}** | {hits[5]} / {n} |",
        "",
        "## Детализация по вопросам",
        "",
        "| # | Вопрос | Ожидаемый документ | Позиция | Top-3 |",
        "|---|---|---|---|---|",
    ]
    for i, r in enumerate(results, 1):
        pos = r["position"] if r["position"] > 0 else "—"
        ok = "✅" if r["top3"] else "❌"
        q_short = r["question"][:60] + ("…" if len(r["question"]) > 60 else "")
        lines.append(f"| {i} | {q_short} | `{r['expected']}` | {pos} | {ok} |")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n📄 Отчёт сохранён: {report_path}")


if __name__ == "__main__":
    evaluate()
