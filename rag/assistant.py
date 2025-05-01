from __future__ import annotations
import logging
from datetime import datetime
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from langchain_core.runnables import RunnablePassthrough
from yandex_cloud_ml_sdk import YCloudML
from typing import List, Dict

from typing import List
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils import get_environment_variables, make_ai_tool
from chat import Chat
from search_engine import SearchEngine
from prompts import (
    QUERY_REFINER_PROMPT,
    PROACTIVE_ASSISTANT_PROMPT,
    PROACTIVE_FRIENDLY_QUESTIONS,
    # SUMMARIZE,
    ASSISTENT_SYSTEM_PROMT,
    ENRICHER,
    REQUEST_TO_RAG,
    ANALYZE_RAG,
    CHECK_ANSWERS
)

# Настройка логгера
def setup_logger():
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"assistant_{timestamp}.log"
    
    logger = logging.getLogger("AssistantLogger")
    logger.setLevel(logging.INFO)
    
    # Явно указываем кодировку UTF-8 при создании файлового обработчика
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()

class Assistant:
    def __init__(self, sdk):
        self.model = sdk.models.completions('yandexgpt').langchain()
        self.rag_engine = SearchEngine(
            data_folder="./data",  # Папка с документами
            sdk=sdk,
            persist_directory="./chroma_db",
            chunk_size=800,  # Можно настроить под наши нужды
            chunk_overlap=100
        )
        self.rag_engine.initialize()
        self.search_tool = DuckDuckGoSearchRun()
        self.chat = Chat(system_prompt=ASSISTENT_SYSTEM_PROMT)
        self.backtrace_question = None
        self.clarifying_questions = None
        
        logger.info("Assistant initialized with model: yandexgpt")

        # Делаем тулзы
        # 1. Очиститель (своеобразный Т9, а также защита от промпт-инъекций)
        self.refiner = make_ai_tool(self.model, QUERY_REFINER_PROMPT)
        logger.info("Refiner tool initialized")

        # 2. Проактивизм. Уточняем запрос
        def _parse_questions(response: str) -> List[str]:
            """Извлекает вопросы из ответа модели"""
            questions = []
            if not response.startswith("None"):
                for line in response.split('\n'):
                    line = line.strip()
                    if line and line[0].isdigit():
                        question = line.split('.', 1)[1].strip()
                        questions.append(question)
            return questions
        self.proactivity = make_ai_tool(self.model, PROACTIVE_ASSISTANT_PROMPT, _parse_questions)
        self.friendly_questions = make_ai_tool(self.model, PROACTIVE_FRIENDLY_QUESTIONS)
        logger.info("Proactivity tools initialized")

        # # 3. Саммаризация. Чтобы корректно работали тулзы и впринципе собирать всю инфу
        # self.summary = make_ai_tool(self.model, SUMMARIZE)
        # logger.info("Summary tool initialized")

        # 3. Обогатитель. Удаляет все ссылки на контекст, чтобы запрос был конкретным
        self.enricher = make_ai_tool(self.model, ENRICHER)

        # 4. Формулироващик поисковых запросов. Универсален и для RAG и для гугла
        self.requester = make_ai_tool(self.model, REQUEST_TO_RAG)

        # 5. Ответ на вопрос.
        self.make_answer = make_ai_tool(self.model, ANALYZE_RAG)

        # 6. Проверка ответов
        self.checker = make_ai_tool(self.model, CHECK_ANSWERS)

    def ask(self, message: str):
        logger.info(f"Received raw message: {message}")
        
        # Очистка сообщения
        refined_message = self.refiner(message)
        logger.info(f"Refined message: {refined_message}")

        # Запись в историю чата
        self.chat.write(refined_message)
        logger.info("Message added to chat history")

        if self.backtrace_question is not None:
            check = self.checker(f"ВОПРОС(Ы): {self.clarifying_questions}\nОТВЕТ: {message}")
            if '[YES]' in check:
                self.chat.write(check, role='assistant')
                self.clarifying_questions = None
                self.chat.write(f"{self.backtrace_question}\n{message}")
            else:
                return self.chat.write(check, role='assistant').content

        # Получение истории чата
        chat_history = self.chat.copy().get_history()
        chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
        logger.info(f"Chat history:\n{chat_history_str}")

        # Саммаризация истории
        # summary_message = self.summary(chat_history_str)
        # self.chat.pop()
        # self.chat.write(summary_message)
        # logger.info(f"Summary generated: {summary_message}")

        # Генерация уточняющих вопросов
        # if self.backtrace_question is None:
        #     questions = '\n'.join(self.proactivity(chat_history_str))
        #     logger.info(f"Generated proactive questions: {questions}")

        #     if questions:
        #         # self.backtrace_question = summary_message
        #         self.backtrace_question = refined_message
        #         self.clarifying_questions = questions
        #         logger.info("Backtrace question set")
                
        #         fr = self.friendly_questions(f'{refined_message}\n{questions}')
        #         logger.info(f"Generated friendly response: {fr}")
        #         return self.chat.write(fr).content
            
        #     logger.info("No proactive questions generated, proceeding to direct response")

        # Удаляем ссылки на контекст
        enriched = self.enricher("\n".join([
            f"{msg.type}: {msg.content}"
            for msg in self.chat[-7:].remove_system_prompt().get_history()
        ]))
        logger.info(f"Generated enriched request: {enriched}")

        # Формируем поисковый запрос
        request = self.requester(enriched)
        logger.info(f"Generated search request: {request}")

        if self.backtrace_question is not None:
            self.chat.pop();
            self.backtrace_question = None

        # Если не нужно производить поиск
        if request == "None":
            response = self.chat.ask(self.model)
            logger.info(f"Generated final response: {response.content}")
            return response.content
        
        # Производим поиск
        docs = '\n\n'.join(self.rag_engine.search(request, n_results=5))
        logger.info(f"Finded docs: {docs}")
        try:
            sites = self.search_tool.run(request, n_results=2)
            logger.info(f"Finded sites: {sites}")
        except DuckDuckGoSearchException as e:
            sites = ''
            logger.info(f"No information was found on the Internet: {e}")
        
        context = f'ДАННЫЕ:\nДОКУМЕНТЫ: {docs}\nСАЙТЫ: {sites}'
        response = self.make_answer(f'{enriched}\n\n{context}')
        logger.info(f"Generated answer: {response}")
        self.chat.write(response, role='assistant')
        return response


if __name__ == '__main__':
    folder_id, api_key = get_environment_variables()

    # Инициализация Yandex Cloud ML
    sdk = YCloudML(folder_id=folder_id, auth=api_key)
    sdk.setup_default_logging()

    print('Подождите, инициализация...', end='\r')
    assistant = Assistant(sdk)
    print(' ' * 27, end='\r')

    while True:
        try:
            message = input('user: ')
            response = assistant.ask(message)
            print(f'assistant: {response}')
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}", exc_info=True)
            # print(f'Error occured: {e}')
            # print('Goodbye!')
            # break
            raise e