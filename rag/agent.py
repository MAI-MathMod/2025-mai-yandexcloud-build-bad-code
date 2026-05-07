from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

try:
    from .admissions_db import AdmissionsDatabase
    from .llm import ChatModel
    from .tools import ToolRegistry
except ImportError:
    from admissions_db import AdmissionsDatabase
    from llm import ChatModel
    from tools import ToolRegistry


SYSTEM_PROMPT = """Ты agentic-ассистент приёмной комиссии МАИ.
Отвечай на языке пользователя, кратко и предметно. Приоритет источников:
1. admissions_sql для проходных баллов, динамики за годы и оценки шансов по ЕГЭ.
2. rag_search для локальной базы знаний, документов, регламентов и информации из чатов.
3. web_search только если локальных данных недостаточно или вопрос общий.

Нельзя выдумывать факты. Если данных не хватает, скажи что именно уточнить.
После инструментов дай финальный ответ обычным текстом, без JSON.
"""


PLANNER_PROMPT = """Выбери инструменты для ответа.
Верни строго JSON одного из видов:
{{"tool_calls":[{{"name":"rag_search","arguments":{{"query":"..."}}}}]}}
{{"answer":"..."}}

Доступные инструменты:
{tool_specs}

SQL schema:
{sql_schema}

История:
{history}

Вопрос пользователя:
{question}
"""


FINAL_PROMPT = """Ответь пользователю на основе результатов инструментов.
Если есть противоречия, укажи источник неопределённости. Не показывай внутренний JSON.

Вопрос:
{question}

TOOL_RESULTS:
{tool_results}
"""


@dataclass
class Conversation:
    messages: list[dict[str, str]] = field(default_factory=list)

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.messages = self.messages[-10:]

    def as_text(self) -> str:
        return "\n".join(f"{item['role']}: {item['content']}" for item in self.messages[-8:])


class AdmissionsAgent:
    def __init__(
        self,
        model: ChatModel,
        tools: ToolRegistry,
        admissions_db: AdmissionsDatabase,
    ):
        self.model = model
        self.tools = tools
        self.admissions_db = admissions_db
        self.conversations: dict[str, Conversation] = {}

    def ask(self, chat_id: str | int, message: str) -> str:
        conversation = self.conversations.setdefault(str(chat_id), Conversation())
        conversation.append("user", message)

        deterministic = self._try_direct_sql_answer(message)
        if deterministic:
            conversation.append("assistant", deterministic)
            return deterministic

        plan = self._plan(conversation, message)
        tool_results = self._execute_plan(plan, message)

        if not tool_results and plan.get("answer"):
            answer = str(plan["answer"])
        else:
            answer = self.model.complete(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": FINAL_PROMPT.format(
                            question=message,
                            tool_results=tool_results or "Инструменты не вызывались.",
                        ),
                    },
                ]
            ).strip()
        conversation.append("assistant", answer)
        return answer

    def _plan(self, conversation: Conversation, question: str) -> dict:
        prompt = PLANNER_PROMPT.format(
            tool_specs=self.tools.specs_for_prompt(),
            sql_schema=self.admissions_db.schema_for_llm(),
            history=conversation.as_text(),
            question=question,
        )
        raw = self.model.complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        return self._parse_json(raw) or self._fallback_plan(question)

    def _execute_plan(self, plan: dict, question: str) -> str:
        calls = plan.get("tool_calls") or []
        if not calls:
            return ""
        results = []
        for call in calls[:4]:
            name = call.get("name")
            arguments = call.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {}
            result = self.tools.call(str(name), arguments)
            results.append(f"## {name}\n{result}")
        return "\n\n".join(results)

    def _fallback_plan(self, question: str) -> dict:
        if self._is_score_question(question):
            total_score = self._extract_total_score(question)
            if total_score:
                return {
                    "tool_calls": [
                        {
                            "name": "admissions_sql",
                            "arguments": {"total_score": total_score, "limit": 10},
                        }
                    ]
                }
        return {"tool_calls": [{"name": "rag_search", "arguments": {"query": question, "limit": 5}}]}

    def _try_direct_sql_answer(self, question: str) -> str | None:
        if not self._is_score_question(question):
            return None
        total_score = self._extract_total_score(question)
        year = self._extract_year(question)
        if total_score is not None and re.search(r"\b(куда|шанс|поступить|пройду|возьмут)\b", question, re.I):
            result = self.tools.call(
                "admissions_sql",
                {"total_score": total_score, "year": year, "limit": 10},
            )
            return (
                f"По базе проходных баллов МАИ за {year or self.admissions_db.latest_year()} год "
                f"при сумме {total_score} можно ориентироваться на такие варианты:\n\n{result}\n\n"
                "Это не гарантия поступления: конкурс, квоты и согласия меняются каждый год."
            )
        program = self._extract_program_query(question)
        if program:
            rows = self.admissions_db.cutoff_history(program)
            if year:
                rows = [row for row in rows if int(row["year"]) == year]
            if rows:
                lines = [
                    f"{row['name']} ({row['code']}), {row['year']}: {row['score']}"
                    for row in rows
                ]
                return "Динамика проходных баллов из локальной SQL-базы:\n\n" + "\n".join(lines)
        return None

    def _parse_json(self, text: str) -> dict | None:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _is_score_question(self, question: str) -> bool:
        return bool(re.search(r"(проходн|балл|егэ|поступ|конкурс|динамик)", question.lower()))

    def _extract_total_score(self, question: str) -> int | None:
        matches = [int(value) for value in re.findall(r"\b([1-3]\d{2})\b", question)]
        plausible = [value for value in matches if 120 <= value <= 310]
        return max(plausible) if plausible else None

    def _extract_year(self, question: str) -> int | None:
        match = re.search(r"\b(20\d{2})\b", question)
        return int(match.group(1)) if match else None

    def _extract_program_query(self, question: str) -> str | None:
        lowered = question.lower()
        aliases = {
            "пми": "прикладная математика",
            "фиит": "фундаментальная информатика",
            "программная инженерия": "программная инженерия",
            "авиастроение": "авиастроение",
            "информатика": "информатика",
        }
        for alias, query in aliases.items():
            if alias in lowered:
                return query
        if "программ" in lowered and "инженер" in lowered:
            return "программная инженерия"
        if "прикладн" in lowered and "математ" in lowered:
            return "прикладная математика"
        if "фундамент" in lowered and "информ" in lowered:
            return "фундаментальная информатика"
        quoted = re.findall(r"[«\"]([^»\"]+)[»\"]", question)
        return quoted[0] if quoted else None
