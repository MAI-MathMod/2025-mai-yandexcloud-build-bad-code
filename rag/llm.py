from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ChatModel(Protocol):
    def plan(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        ...

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        ...


@dataclass
class LangChainYandexGPTModel:
    folder_id: str
    api_key: str
    model_name: str = "yandexgpt-5-pro/latest"
    base_url: str = "https://ai.api.cloud.yandex.net/v1"
    max_output_tokens: int = 1500

    def __post_init__(self) -> None:
        from langchain_openai import ChatOpenAI

        model = self.model_name
        if not model.startswith("gpt://"):
            model = f"gpt://{self.folder_id}/{model}"
        self._model = ChatOpenAI(
            model=model,
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={"OpenAI-Project": self.folder_id},
            max_tokens=self.max_output_tokens,
        )

    def plan(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        response = self._model.bind_tools(tools, tool_choice="auto").bind(temperature=temperature).invoke(
            self._convert_messages(messages)
        )
        tool_calls = [
            {
                "name": call.get("name", ""),
                "arguments": call.get("args") or {},
                "id": call.get("id"),
            }
            for call in response.tool_calls
        ]
        if tool_calls:
            return {"tool_calls": tool_calls}
        return {"answer": self._content_to_text(response.content)}

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        structured_model = self._model.with_structured_output(
            schema,
            method="json_schema",
            strict=True,
        )
        result = structured_model.bind(temperature=temperature).invoke(
            self._convert_messages(messages)
        )
        if isinstance(result, dict):
            return result
        if hasattr(result, "model_dump"):
            return result.model_dump()
        raise TypeError("Structured model returned an unsupported result")

    def _convert_messages(self, messages: list[dict[str, str]]) -> list[Any]:
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
        return converted

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(block.get("text", "")) if isinstance(block, dict) else str(block)
                for block in content
            ).strip()
        return str(content)


class OfflineAdmissionsModel:
    """Small fallback used when cloud credentials are not configured."""

    def plan(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        last = messages[-1]["content"] if messages else ""
        return {
            "tool_calls": [
                {"name": "rag_search", "arguments": {"query": last[:300]}}
            ]
        }

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        last = messages[-1]["content"] if messages else ""
        return {"answer": self._answer_from_tool_results(last)}

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
