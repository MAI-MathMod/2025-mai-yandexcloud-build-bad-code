from __future__ import annotations

import os
from typing import Dict, List, Literal, Optional, Union
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.base import BaseMessage

try:
    from .utils import get_environment_variables
except ImportError:
    from utils import get_environment_variables


class Chat:
    def __init__(self, system_prompt: Optional[str] = None):
        self.has_system = system_prompt is not None
        if system_prompt is not None:
            self.history = [SystemMessage(content=system_prompt)]
        else:
            self.history = []
        
    def change_system_prompt(self, new_system_prompt: str):
        if self.has_system:
            self.history[0] = SystemMessage(content=new_system_prompt)
        else:
            self.has_system = True
            self.history = [SystemMessage(content=new_system_prompt)] + self.history
        return self

    def copy(self):
        copy = Chat()
        copy.history = self.history
        copy.has_system = self.has_system
        return copy
    
    def remove_system_prompt(self):
        if self.has_system and self.history:
            self.history = self.history[1:]
            self.has_system = False
        return self
    
    def write(self, msg: str, role: Literal['user', 'assistant'] = 'user'):
        if role == 'user':
            added = HumanMessage(content=msg)
            self.history.append(added)
        else:
            added = AIMessage(content=msg)
            self.history.append(added)
        return added
    
    def pop(self):
        return self.history.pop()

    def get_history(self):
        return self.history

    def ask(self, model):
        ans = model.invoke(self.get_history())
        self.write(ans.content, role='assistant')
        return ans

    def __getitem__(self, key: Union[int, slice]) -> Union[BaseMessage, List[BaseMessage]]:
        if isinstance(key, int):
            return self.history[key]
        else:
            copy = Chat()
            copy.history = self.history[key]
            if copy.history and copy.history[0].type == 'system':
                copy.has_system = True
            return copy

if __name__ == '__main__':
    from langchain_openai import ChatOpenAI

    folder_id, api_key = get_environment_variables()

    chat = Chat('Ты - работник приёмной комиссии Московского Авиационного института, помогаешь студентам с вопросами по поступлению')
    chat.write('Привет, что расскажешь про ваш вуз?')

    print('asking...')
    model_name = os.getenv("YC_MODEL_NAME", "yandexgpt-5-pro/latest")
    if not model_name.startswith("gpt://"):
        model_name = f"gpt://{folder_id}/{model_name}"
    model = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        base_url="https://ai.api.cloud.yandex.net/v1",
        default_headers={"OpenAI-Project": folder_id},
        max_tokens=1500,
    )
    print(chat.ask(model))
