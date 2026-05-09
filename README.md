# Agentic-ассистент для приёмной комиссии МАИ

RAG · Function Calling · Yandex GPT 5 Pro · jina-embeddings-v3 · Qdrant

Проект переделан из монолитного RAG-бота в агентного ассистента для абитуриентов МАИ. Ассистент отвечает на вопросы о поступлении, документах, общежитии, льготах и проходных баллах, а для расчёта шансов использует SQL-инструмент поверх локальной базы направлений.

## Что внутри

- `rag/assistant.py` — совместимый фасад `Assistant.ask(chat_id, message)` для очередей и локального запуска.
- `rag/agent.py` — агентный цикл: планирование tool calls, выполнение инструментов, финальный ответ.
- `rag/tools.py` — реестр инструментов: `rag_search`, `admissions_sql`, `web_search`.
- `rag/knowledge_base.py` — очистка данных, удаление ПДн/ненормативной лексики, дедупликация и чанкинг.
- `rag/retrieval.py` — гибридный retrieval: Qdrant/vector search + BM25-like lexical search.
- `rag/embeddings.py` — `jina-embeddings-v3`; без `JINA_API_KEY` включается детерминированный offline fallback.
- `rag/admissions_db.py` — SQLite-база направлений и проходных баллов из `rag/data/cutoff_points/*.csv`.
- `rag/chat_mining.py` — пайплайн для выгрузок чатов: топ-200 вопросов и QA-датасет до 500 пар.
- `rag/evaluation.py` — каркас end-to-end оценки на QA-датасете; RAGAS можно подключить через зависимости.

## Данные

База знаний лежит в `rag/data` и состоит из:

- страниц приёмной комиссии;
- нормативных документов;
- локальных заметок из чатов;
- таблиц проходных баллов за 2018-2024 годы;
- whitelist доменов для обновления источников: `rag/data/source_whitelist.json`.

Таблицы не индексируются как сырой CSV. Строки превращаются в структурированный текст с контекстными заголовками: год, код направления, конкурсная группа, проходной балл. Это улучшает retrieval по запросам вроде "ПМИ проходной 2024" и даёт SQL-инструменту нормальную табличную базу.

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r rag/requirements.txt
python -m rag.cli build-db
python -m rag.cli rebuild-index
python -m rag.assistant
```

Локальный интерактивный формат:

```text
user-1|У меня 260 баллов ЕГЭ. Куда я могу пройти?
```

## Переменные окружения

```env
YC_FOLDER_ID=...
YC_API_KEY=...
YC_MODEL_NAME=yandexgpt-5-pro/latest
JINA_API_KEY=...
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=mai_admissions_kb
ENABLE_WEB_SEARCH=1
AUTO_INDEX=1
```

Старые имена `folder_id` и `api_key` тоже поддерживаются.

## Docker

```bash
docker compose up --build
```

Compose поднимает:

- Go Telegram bot gateway;
- Python `rag-worker`;
- Qdrant.

## Function Calling

Агент передаёт схемы инструментов в нативный `ChatOpenAI.bind_tools()` и читает
вызовы из `AIMessage.tool_calls`:

- `rag_search` — локальный RAG по очищенному индексу;
- `admissions_sql` — безопасный `SELECT` к SQLite или готовый расчёт по `total_score`;
- `web_search` — fallback в интернет после локального поиска.

Финальный ответ генерируется через `with_structured_output(..., method="json_schema")`
по строгой схеме с полем `answer`; пользователю возвращается только содержимое поля.

Для вопросов о проходных баллах есть детерминированный fast path: ассистент сразу вызывает SQL-инструмент, не заставляя модель угадывать численные данные.

## Оценка качества

Пайплайн поддерживает два уровня:

- компонентная оценка retrieval/generation через RAGAS: `context_recall`,
  `faithfulness`, `factual_correctness`;
- end-to-end прогон агента по QA-датасету. `ragas_score` считается как
  геометрическое среднее трёх метрик, точность — как доля примеров с
  `factual_correctness >= 0.70`.

Оценка использует отдельную judge-модель. Fallback отсутствует: без ключей или при
ошибке judge-модели команда завершается с ошибкой.

```bash
export RAGAS_API_KEY=...
export RAGAS_JUDGE_MODEL=gpt-4o-mini
# Нужны также YC_FOLDER_ID, YC_API_KEY, JINA_API_KEY и доступный Qdrant.
python -m rag.cli evaluate rag/data/evaluation/qa_dataset.sample.jsonl \
  --output rag/data/evaluation/ragas_report.json
```

Для OpenAI-совместимого провайдера можно задать `RAGAS_BASE_URL`. Порог точности
настраивается через `RAGAS_FACTUAL_CORRECTNESS_THRESHOLD`.

Пример датасета лежит в `rag/data/evaluation/qa_dataset.sample.jsonl`.

## Команды для данных

```bash
python -m rag.cli build-db
python -m rag.cli rebuild-index
python -m rag.cli update-index
python -m rag.cli mine-chats /path/to/chat_exports rag/data/from_chat/derived --top-n 200 --dataset-size 500
```

`update-index` сравнивает хэши файлов с `.index_state.json` и переиндексирует только изменённые источники.

## Результаты целевой конфигурации

Числа не зашиты в README. Команда `evaluate` формирует JSON-отчёт с итоговым
RAGAS score, каждой метрикой, точностью, задержкой и результатами всех примеров.
Так результат привязан к конкретному датасету и текущей облачной конфигурации.
