from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
        ...


@dataclass
class YandexGPTModel:
    sdk: object
    model_name: str = "yandexgpt-5-pro"

    def __post_init__(self) -> None:
        self._model = self.sdk.models.completions(self.model_name).langchain()

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

            converted = []
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "system":
                    converted.append(SystemMessage(content=content))
                elif role == "assistant":
                    converted.append(AIMessage(content=content))
                else:
                    converted.append(HumanMessage(content=content))
            return str(self._model.invoke(converted).content)
        except Exception:
            prompt = "\n\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
            return str(self._model.invoke(prompt).content)


class OfflineAdmissionsModel:
    """Small fallback used when cloud credentials are not configured."""

    def complete(self, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
        last = messages[-1]["content"] if messages else ""
        if "TOOL_RESULTS" in last:
            return self._answer_from_tool_results(last)
        return (
            '{"tool_calls":[{"name":"rag_search","arguments":{"query":"'
            + last.replace('"', '\\"')[:300]
            + '"}}]}'
        )

    def _answer_from_tool_results(self, prompt: str) -> str:
        marker = "TOOL_RESULTS:"
        if marker not in prompt:
            return "Я могу помочь с поступлением в МАИ: документами, сроками, проходными баллами и направлениями."
        content = prompt.split(marker, 1)[1].strip()
        return (
            "Нашёл релевантные данные в базе. Коротко по вопросу:\n\n"
            f"{content[:1800]}\n\n"
            "Если нужны шансы по ЕГЭ, напишите сумму баллов и интересующее направление."
        )

