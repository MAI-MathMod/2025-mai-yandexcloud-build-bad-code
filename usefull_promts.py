# Standard library imports
import os
import json
import re
from typing import List, Optional

# Third-party imports
from yandex_cloud_ml_sdk import YCloudML

# Local imports
from utils import get_environment_variables
from prompts import TOPICS_EXTRACTOR_PROMPT, QUERY_REFINER_PROMPT, PROACTIVE_ASSISTANT_PROMPT

# Global variables
folder_id, api_key = get_environment_variables()
sdk = YCloudML(folder_id=folder_id, auth=api_key)
sdk.setup_default_logging(log_level='DEBUG')

question_topics = """
1. Проходные баллы и статистика:
   - Проходные баллы прошлых лет
   - Текущее распределение баллов подавших документы
   - Минимальные баллы для подачи документов
   - Текущий предполагаемый проходной балл

2. Учебный процесс:
   - Предметы и их содержание
   - Расписание занятий
   - Формат обучения (очное/заочное)
   - Длительность обучения
   - Практики и стажировки
   - В каких корпусах проходят занятия

3. Поступление:
   - Сроки подачи документов
   - Необходимые документы
   - Вступительные испытания
   - Особые условия поступления (олимпиады, целевое обучение)
   - Количество бюджетных мест

4. Студенческая жизнь:
   - Общежитие (расположение, блочная/коридорная/секционная, количество в комнате)
   - Стипендии и материальная поддержка
   - Внеучебная деятельность
   - Спортивные секции
   - Студенческие организации

5. Карьерные перспективы:
   - Трудоустройство после выпуска
   - Возможность совмещать работу и учёбу
   - Партнерские компании
   - Средняя зарплата выпускников
   - Возможности для карьерного роста

6. Инфраструктура:
   - Расположение корпусов
   - Расположение общежитий
   - Библиотека
   - Компьютерные классы
   - Коворкинги
   - Спортивные сооружения
   - Столовые и кафе

7. Международное сотрудничество:
   - Программы обмена
   - Стажировки за рубежом
   - Двойные дипломы
   - Международные проекты

8. Дополнительные возможности:
   - Курсы повышения квалификации
   - Дополнительное образование
   - Научная деятельность
   - Стартап-инкубаторы
"""

# Classes
class TopicsExtractor:
    def __init__(self, question_topics):
        self.model = sdk.models.completions('yandexgpt', model_version='rc')
        model.configure(temperature=0.3, response_format='json')
        self.system_prompt = TOPICS_EXTRACTOR_PROMPT.format(question_topics=question_topics)
    
    def __call__(self, user_message: str) -> list:
        """
        Анализирует сообщение пользователя и возвращает найденные темы

        :param user_message: Текст запроса пользователя
        :return: Список найденных тем или None при ошибке
        """
        try:
            result = self.model.configure(
                response_format='json'
            ).run([
                {'role': 'system', 'text': self.system_prompt},
                {'role': 'user', 'text': user_message}
            ])
            result = self._clean_json_response(result.text)
            return json.loads(result) if result else []

        except (json.JSONDecodeError, AttributeError, KeyError) as e:
            print(f"Ошибка обработки ответа: {e}")
            return None

    def _clean_json_response(self, raw_response: str) -> str:
        try:
            # Пробуем распарсить как есть
            json.loads(raw_response)
            return raw_response
        except json.JSONDecodeError:
            # Если не получается, ищем JSON-подобную структуру
            match = re.search(r'(\[.*?\])', raw_response)
            return match.group(1) if match else '[]'


class QueryRefiner:
    def __init__(self):
        """
        Инициализация уточняющего преобразователя запросов
        
        :param folder_id: Идентификатор каталога Yandex Cloud
        """
        self.model = sdk.models.completions('yandexgpt', model_version='rc')
        self.system_prompt = QUERY_REFINER_PROMPT

    def __call__(self, query: str) -> Optional[str]:
        """
        Уточняет и исправляет пользовательский запрос
        
        :param query: Исходный запрос пользователя
        :return: Уточненный запрос или None при ошибке
        """
        try:
            response = self.model.configure(
                temperature=0.1,
                max_tokens=500
            ).run([
                {'role': 'system', 'text': self.system_prompt},
                {'role': 'user', 'text': query}
            ])
            
            return self._clean_response(response[0].text)
            
        except Exception as e:
            print(f"Ошибка обработки запроса: {e}")
            return None
    
    def _clean_response(self, response: str) -> str:
        """Очищает ответ модели от возможного мусора"""
        # Удаляем кавычки если они есть
        cleaned = response.strip('"\'')
        # Удаляем возможные пояснения после запроса
        return re.split(r'\n|\.\.\.', cleaned)[0]


class ProactiveAssistant:
    def __init__(self):
        """
        Инициализация проактивного помощника
        
        :param folder_id: Идентификатор каталога Yandex Cloud
        """
        self.model = sdk.models.completions('yandexgpt', model_version='rc')
        self.system_prompt = PROACTIVE_ASSISTANT_PROMPT
    
    def __call__(self, query: str) -> Optional[List[str]]:
        """
        Анализирует запрос и возвращает список уточняющих вопросов
        
        :param query: Запрос пользователя
        :return: Список уточняющих вопросов или None при ошибке
        """
        try:
            response = self.model.configure(
                temperature=0.3,
                max_tokens=300
            ).run([
                {'role': 'system', 'text': self.system_prompt},
                {'role': 'user', 'text': query}
            ])
            
            return self._parse_questions(response[0].text)
            
        except Exception as e:
            print(f"Ошибка обработки запроса: {e}")
            return None
    
    def _parse_questions(self, response: str) -> List[str]:
        """Извлекает вопросы из ответа модели"""
        questions = []
        for line in response.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                question = line.split('.', 1)[1].strip()
                questions.append(question)
        return questions