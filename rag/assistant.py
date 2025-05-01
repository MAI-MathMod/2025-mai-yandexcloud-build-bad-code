from __future__ import annotations
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from langchain_core.runnables import RunnablePassthrough
from yandex_cloud_ml_sdk import YCloudML
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
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()

class Assistant:
    def __init__(self, sdk):
        self.model = sdk.models.completions('yandexgpt').langchain()
        self.rag_engine = SearchEngine(
            data_folder="./data",
            sdk=sdk,
            persist_directory="./chroma_db",
            chunk_size=800,
            chunk_overlap=100
        )
        self.rag_engine.initialize()
        self.search_tool = DuckDuckGoSearchRun()
        self.active_chats: Dict[str, Chat] = {}  # Словарь для хранения чатов по chat_id
        logger.info("Assistant initialized with model: yandexgpt")

        # Инициализация инструментов
        def postprocess_refiner(msg):
            if msg == 'В интернете есть много сайтов с информацией на эту тему. [Посмотрите, что нашлось в поиске](https://ya.ru)':
                return 'Я хочу поступить в Бауманку'
            return msg

        self.refiner = make_ai_tool(self.model, QUERY_REFINER_PROMPT, postprocess_refiner)
        
        def _parse_questions(response: str) -> List[str]:
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
        self.enricher = make_ai_tool(self.model, ENRICHER)
        self.requester = make_ai_tool(self.model, REQUEST_TO_RAG)
        self.make_answer = make_ai_tool(self.model, ANALYZE_RAG)
        self.checker = make_ai_tool(self.model, CHECK_ANSWERS)
        
        logger.info("All tools initialized")

    def create_chat(self, chat_id: str) -> Chat:
        """Создает новый чат для указанного chat_id"""
        if chat_id not in self.active_chats:
            self.active_chats[chat_id] = Chat(system_prompt=ASSISTENT_SYSTEM_PROMT)
            logger.info(f"Created new chat for chat_id: {chat_id}")
        return chat_id

    # def ask(self, chat_id: str, message: str) -> str:
    #     """Обрабатывает запрос пользователя в указанном чате"""
    #     logger.info(f"Received message from chat_id {chat_id}: {message[:50]}...")
        
    #     # Получаем или создаем чат
    #     chat = self.active_chats.get(chat_id)
    #     if chat is None:
    #         chat = self.create_chat(chat_id)
        
    #     # Очистка сообщения
    #     refined_message = self.refiner(message)
    #     logger.info(f"Refined message: {refined_message}")

    #     # Запись в историю чата
    #     chat.write(refined_message)
    #     logger.info("Message added to chat history")

    #     # Проверка уточняющих вопросов
    #     if hasattr(chat, 'backtrace_question') and chat.backtrace_question is not None:
    #         check = self.checker(f"ВОПРОС(Ы): {chat.clarifying_questions}\nОТВЕТ: {message}")
    #         if '[YES]' in check:
    #             chat.write(check, role='assistant')
    #             chat.clarifying_questions = None
    #             chat.write(f"{chat.backtrace_question}\n{message}")
    #         else:
    #             return chat.write(check, role='assistant').content

    #     # Получение истории чата
    #     chat_history = chat.copy().get_history()
    #     chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    #     # Удаляем ссылки на контекст
    #     enriched = self.enricher("\n".join([
    #         f"{msg.type}: {msg.content}"
    #         for msg in chat[-7:].remove_system_prompt().get_history()
    #     ]))
    #     logger.info(f"Generated enriched request: {enriched}")

    #     # Формируем поисковый запрос
    #     request = self.requester(enriched)
    #     logger.info(f"Generated search request: {request}")

    #     if hasattr(chat, 'backtrace_question') and chat.backtrace_question is not None:
    #         chat.pop()
    #         chat.backtrace_question = None

    #     # Если не нужно производить поиск
    #     if request == "None":
    #         response = chat.ask(self.model)
    #         logger.info(f"Generated final response: {response.content}")
    #         return response.content
        
    #     # Производим поиск
    #     docs = '\n\n'.join(self.rag_engine.search(request, n_results=5))
    #     logger.info(f"Finded docs: {docs[:100]}...")
    #     try:
    #         sites = self.search_tool.run(request, n_results=2)
    #         logger.info(f"Finded sites: {sites}")
    #     except DuckDuckGoSearchException as e:
    #         sites = ''
    #         logger.info(f"No information was found on the Internet: {e}")
        
    #     context = f'ДАННЫЕ:\nДОКУМЕНТЫ: {docs}\nСАЙТЫ: {sites}'
    #     response = self.make_answer(f'{enriched}\n\n{context}')
    #     logger.info(f"Generated answer: {response}")
    #     chat.write(response, role='assistant')
    #     return response

    def ask(self, chat_id: str, message: str) -> str:
        logger.info(f"Received raw message: {message}")

        # Получаем или создаем чат
        chat = self.active_chats.get(chat_id)
        if chat is None:
            self.create_chat(chat_id)
            chat = self.active_chats.get(chat_id)
        
        # Очистка сообщения
        refined_message = self.refiner(message)
        logger.info(f"Refined message: {refined_message}")

        # Запись в историю чата
        chat.write(refined_message)
        logger.info("Message added to chat history")

        # if self.backtrace_question is not None:
        #     check = self.checker(f"ВОПРОС(Ы): {self.clarifying_questions}\nОТВЕТ: {message}")
        #     if '[YES]' in check:
        #         chat.write(check, role='assistant')
        #         self.clarifying_questions = None
        #         chat.write(f"{self.backtrace_question}\n{message}")
        #     else:
        #         return chat.write(check, role='assistant').content

        # Получение истории чата
        chat_history = chat.copy().get_history()
        chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
        logger.info(f"Chat history:\n{chat_history_str}")

        # Саммаризация истории
        # summary_message = self.summary(chat_history_str)
        # chat.pop()
        # chat.write(summary_message)
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
        #         return chat.write(fr).content
            
        #     logger.info("No proactive questions generated, proceeding to direct response")

        # Удаляем ссылки на контекст
        enriched = self.enricher("\n".join([
            f"{msg.type}: {msg.content}"
            for msg in chat[-7:].remove_system_prompt().get_history()
        ]))
        logger.info(f"Generated enriched request: {enriched}")

        # Формируем поисковый запрос
        request = self.requester(enriched)
        logger.info(f"Generated search request: {request}")

        # if self.backtrace_question is not None:
        #     chat.pop();
        #     self.backtrace_question = None

        # Если не нужно производить поиск
        if request == "None":
            response = chat.ask(self.model)
            logger.info(f"Generated final response: {response.content}")
            return response.content
        
        # Производим поиск
        # docs = self.ask_with_refinement(chat_id, request)
        docs = '\n---\n'.join(self.rerank_documents(enriched, self.rag_engine.search(request, n_results=8)))
        logger.info(f"Finded docs: {docs}")
        
        # try:
            # sites = self.search_tool.run(request, n_results=3)
            # logger.info(f"Finded sites: {sites}")
        # except DuckDuckGoSearchException as e:
        #     sites = ''
        #     logger.info(f"No information was found on the Internet: {e}")
        
        # context = f'ДАННЫЕ:\n{docs}\n{sites}'
        context = f'КОНТЕКСТ:\n\n{docs}'
        response = self.make_answer(f'{enriched}\n\n{context}')
        logger.info(f"Generated answer: {response}")
        chat.write(response, role='assistant')
        return response

    def rerank_documents(self, question: str, documents: list[str], top_n: int = 3) -> list[str]:
        """
        Реранжирует документы по релевантности вопросу
        :param question: текущий вопрос
        :param documents: список документов
        :param top_n: количество возвращаемых документов
        :return: топ-N наиболее релевантных документов
        """
        ranked = []
        for doc in documents:
            prompt = f"""Оцени релевантность документа вопросу от 0 до 100. Ответь только числом.
            
            Вопрос: {question}
            Документ: {doc}
            """
            try:
                current_response = self.model.invoke(prompt).content
                numbers = re.findall(r'\d+', current_response)
                score = int(numbers[0]) if numbers else 0
                ranked.append((score, doc))
            except Exception as e:
                print(e)
                continue
        
        # Сортируем по убыванию оценки и берем топ-N
        ranked.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in ranked[:top_n]]

if __name__ == '__main__':
    folder_id, api_key = get_environment_variables()

    # Инициализация Yandex Cloud ML
    sdk = YCloudML(folder_id=folder_id, auth=api_key)
    sdk.setup_default_logging()

    print('Подождите, инициализация...', end='\r')
    assistant = Assistant(sdk)
    print(' ' * 27, end='\r')

    # Пример использования с разными чатами
    chat_id1 = "user123"
    chat_id2 = "user456"

    # Создаем чаты
    assistant.create_chat(chat_id1)
    assistant.create_chat(chat_id2)

    while True:
        try:
            # Эмулируем запросы от разных пользователей
            user_input = input('Введите chat_id и сообщение (формат: chat_id|message): ')
            if '|' not in user_input:
                print("Неверный формат. Используйте: chat_id|message")
                continue
                
            chat_id, message = user_input.split('|', 1)
            response = assistant.ask(chat_id, message)
            print(f'assistant ({chat_id}): {response}')
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}", exc_info=True)
            print(f'Error occurred: {e}')
            raise e
            break