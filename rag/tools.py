from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

try:
    from .admissions_db import AdmissionsDatabase, format_chances
    from .retrieval import HybridRetriever
except ImportError:
    from admissions_db import AdmissionsDatabase, format_chances
    from retrieval import HybridRetriever


ToolHandler = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {tool.name: tool for tool in tools}

    def specs_for_prompt(self) -> str:
        specs = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]
        return json.dumps(specs, ensure_ascii=False, indent=2)

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Инструмент {name!r} не найден."
        try:
            return self._tools[name].handler(arguments)
        except Exception as exc:
            return f"Ошибка инструмента {name}: {exc}"


def build_default_tools(
    retriever: HybridRetriever | None,
    admissions_db: AdmissionsDatabase,
    enable_web_search: bool = True,
) -> ToolRegistry:
    def rag_search(arguments: dict[str, Any]) -> str:
        if retriever is None:
            return "RAG-индекс ещё не собран."
        query = str(arguments.get("query", "")).strip()
        limit = int(arguments.get("limit", 5))
        results = retriever.search(query, limit=limit)
        if not results:
            return "В локальной базе знаний ничего не найдено."
        blocks = []
        for item in results:
            source = item.metadata.get("source", "unknown")
            blocks.append(f"[source={source}; score={item.score:.3f}]\n{item.text}")
        return "\n\n---\n\n".join(blocks)

    def admissions_sql(arguments: dict[str, Any]) -> str:
        if "total_score" in arguments:
            year = arguments.get("year")
            chances = admissions_db.eligible_programs(
                total_score=int(arguments["total_score"]),
                year=int(year) if year else None,
                limit=int(arguments.get("limit", 10)),
            )
            return format_chances(chances)
        query = str(arguments.get("query", "")).strip()
        rows = admissions_db.execute_select(query)
        return json.dumps(rows[:50], ensure_ascii=False, indent=2)

    def web_search(arguments: dict[str, Any]) -> str:
        if not enable_web_search:
            return "Веб-поиск отключён конфигурацией."
        query = str(arguments.get("query", "")).strip()
        whitelist = arguments.get("whitelist_domains") or []
        return _duckduckgo_search(query, whitelist=whitelist)

    return ToolRegistry(
        [
            Tool(
                name="rag_search",
                description="Ищет в локальной базе знаний МАИ, очищенной от ПДн и разложенной на чанки.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
                handler=rag_search,
            ),
            Tool(
                name="admissions_sql",
                description=(
                    "Выполняет безопасный SELECT к базе направлений и проходных баллов. "
                    "Можно передать total_score/year для готового расчёта шансов."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "total_score": {"type": "integer"},
                        "year": {"type": "integer"},
                        "limit": {"type": "integer", "default": 10},
                    },
                },
                handler=admissions_sql,
            ),
            Tool(
                name="web_search",
                description="Ищет в интернете, когда RAG не дал ответа или вопрос общий.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "whitelist_domains": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["query"],
                },
                handler=web_search,
            ),
        ]
    )


def _duckduckgo_search(query: str, whitelist: list[str] | None = None) -> str:
    try:
        from duckduckgo_search import DDGS
    except Exception:
        return "Пакет duckduckgo_search не установлен."

    if whitelist:
        domain_filter = " OR ".join(f"site:{domain}" for domain in whitelist)
        query = f"({domain_filter}) {query}"
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
    except Exception as exc:
        return f"Веб-поиск недоступен: {exc}"
    if not results:
        return "В интернете ничего не найдено."
    lines = []
    for result in results:
        title = re.sub(r"\s+", " ", result.get("title", "")).strip()
        href = result.get("href", "")
        body = re.sub(r"\s+", " ", result.get("body", "")).strip()
        lines.append(f"{title}\n{href}\n{body}")
    return "\n\n".join(lines)
