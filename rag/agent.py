from __future__ import annotations

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
Финальный ответ возвращай в поле answer заданной JSON Schema. Сам текст answer — обычный ответ пользователю.
"""


PLANNER_PROMPT = """Выбери подходящие инструменты через нативный tool calling.
Если инструмент не нужен, дай черновик ответа обычным текстом.

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


FINAL_ANSWER_SCHEMA = {
    "title": "AdmissionsFinalAnswer",
    "description": "Финальный ответ ассистента приёмной комиссии пользователю.",
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "Готовый ответ пользователю без служебных данных и JSON.",
        }
    },
    "required": ["answer"],
    "additionalProperties": False,
}


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

        plan = self._plan(conversation, message)
        tool_results = self._execute_plan(plan)
        context = tool_results or str(plan.get("answer") or "Инструменты не вызывались.")
        structured = self.model.complete_structured(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": FINAL_PROMPT.format(
                        question=message,
                        tool_results=context,
                    ),
                },
            ],
            FINAL_ANSWER_SCHEMA,
        )
        answer = str(structured.get("answer", "")).strip()
        if not answer:
            answer = "Не удалось сформировать ответ. Попробуйте уточнить вопрос."
        conversation.append("assistant", answer)
        return answer

    def _plan(self, conversation: Conversation, question: str) -> dict:
        prompt = PLANNER_PROMPT.format(
            sql_schema=self.admissions_db.schema_for_llm(),
            history=conversation.as_text(),
            question=question,
        )
        return self.model.plan(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            self.tools.schemas_for_model(),
        )

    def _execute_plan(self, plan: dict) -> str:
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
