from __future__ import annotations

from yandex_cloud_ml_sdk import YCloudML
from typing import List, Dict, Literal, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
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
    folder_id, api_key = get_environment_variables()

    # Инициализация Yandex Cloud ML
    sdk = YCloudML(folder_id=folder_id, auth=api_key)
    sdk.setup_default_logging()

    chat = Chat('Ты - работник приёмной комиссии Московского Авиационного института, помогаешь студентам с вопросами по поступлению')
    chat.write('Привет, что расскажешь про ваш вуз?')

    print('asking...')
    model = sdk.models.completions('yandexgpt').langchain()
    print(chat.ask(model))