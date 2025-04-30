import os
from typing import List, Tuple, Dict, Optional
import random
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime
import time
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataGenerator:
    """Класс для генерации синтетических тестовых данных для оценки RAG-системы"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "test_data"):
        """
        Инициализация генератора данных
        
        Args:
            data_dir: путь к директории с данными
            output_dir: путь к директории для сохранения тестовых данных
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        load_dotenv()
        
        # Создаем директорию для тестовых данных, если её нет
        os.makedirs(output_dir, exist_ok=True)
        
        # Инициализация клиента OpenAI для OpenRouter
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
    def get_all_files(self) -> List[str]:
        """Рекурсивно получает все файлы из директории data"""
        files = []
        for root, _, filenames in os.walk(self.data_dir):
            for filename in filenames:
                if filename.endswith('.txt'):
                    files.append(os.path.join(root, filename))
        return files
    
    def read_file_content(self, file_path: str) -> str:
        """Читает содержимое файла"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def extract_facts(self) -> List[str]:
        """Извлекает случайные факты из базы знаний
        
        Returns:
            Список случайных фактов (от 2 до 4 файлов)
        """
        # Получаем все файлы из базы знаний
        files = self.get_all_files()
        if not files:
            raise ValueError("No data files found in the specified directory")
            
        # Выбираем случайное количество файлов от 2 до 4
        num_facts = random.randint(2, 4)
        selected_files = random.sample(files, min(num_facts, len(files)))
        
        facts = []
        for file_path in selected_files:
            content = self.read_file_content(file_path)
            # Если файл длиннее 3000 символов, берем случайный срез
            if len(content) > 3000:
                start = random.randint(0, len(content) - 3000)
                content = content[start:start + 3000]
            facts.append(content)
            
        return facts
    
    def _make_api_call(self, model: str, messages: List[Dict], temperature: float, max_retries: int = 5) -> Optional[str]:
        """
        Выполняет API-запрос с повторными попытками и экспоненциальной задержкой
        
        Args:
            model: название модели
            messages: список сообщений
            temperature: параметр температуры
            max_retries: максимальное количество попыток
            
        Returns:
            Ответ от API или None в случае неудачи
        """
        base_delay = 1  # начальная задержка в секундах
        max_delay = 60  # максимальная задержка в секундах
        
        for attempt in range(max_retries):
            try:
                # Добавляем логирование запроса
                logger.info(f"Отправка запроса к API (попытка {attempt + 1}/{max_retries}):")
                logger.info(f"Model: {model}")
                logger.info(f"Messages: {messages}")
                
                completion = self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "https://github.com/your-repo",
                        "X-Title": "RAG Evaluation",
                    },
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1000,  # Добавляем ограничение на длину ответа
                    timeout=30  # Добавляем таймаут
                )
                
                # Добавляем логирование ответа
                logger.info(f"Получен ответ от API: {completion}")
                
                # Проверяем, что completion и его атрибуты не None
                if not completion or not completion.choices or len(completion.choices) == 0:
                    raise ValueError("Empty response from API")
                    
                message = completion.choices[0].message
                if not message or not message.content:
                    raise ValueError("Empty message content in API response")
                    
                content = message.content.strip()
                if not content:
                    raise ValueError("Empty content after stripping")
                    
                logger.info(f"Успешно получен ответ: {content[:100]}...")
                return content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Ошибка API после {max_retries} попыток: {str(e)}")
                    return None
                
                # Вычисляем задержку с экспоненциальным ростом
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(f"Попытка {attempt + 1}/{max_retries} не удалась: {str(e)}. Повторная попытка через {delay} секунд...")
                time.sleep(delay)

    def generate_query(self, facts: List[str]) -> str:
        """Генерирует запрос на основе фактов"""
        prompt = f"""Сгенерируй естественный вопрос, ответ на который содержится в предложенных фактах. Вопрос должен относиться к теме поступления в МАИ, поэтому некоторые факты придётся игнорировать. В твоём ответе должен быть только вопрос и ничего более.

        Факты:
        {chr(10).join(facts)}
        """

        messages = [{"role": "user", "content": prompt}]
        response = self._make_api_call(
            model="qwen/qwen3-30b-a3b:free",
            messages=messages,
            temperature=0.7
        )
        
        if response is None:
            # В случае неудачи генерируем простой вопрос
            logger.warning("Не удалось сгенерировать вопрос через API. Используем простой вопрос.")
            return f"Что можно сказать о {facts[0].split()[0]}?"
            
        return response
    
    def generate_golden_answer(self, facts: List[str], query: str) -> str:
        """Генерирует эталонный ответ с помощью OpenRouter"""
        prompt = f"""На основе следующих фактов дай полный и точный ответ на вопрос:
        
        Вопрос: {query}
        
        Факты:
        {chr(10).join(facts)}
        
        Ответ должен быть точным, полным и основанным только на предоставленных фактах."""
        
        messages = [{"role": "user", "content": prompt}]
        response = self._make_api_call(
            model="qwen/qwen3-30b-a3b:free",
            messages=messages,
            temperature=0.3
        )
        
        if response is None:
            # В случае неудачи используем конкатенацию фактов
            logger.warning("Не удалось сгенерировать ответ через API. Используем конкатенацию фактов.")
            return " ".join(facts)
            
        return response
    
    def save_test_data(self, test_queries: List[str], expected_contexts: List[Tuple[str]], golden_answers: List[str]) -> str:
        """
        Сохраняет тестовые данные в JSON файл
        
        Args:
            test_queries: список тестовых запросов
            expected_contexts: список кортежей с эталонными контекстами
            golden_answers: список эталонных ответов
            
        Returns:
            Путь к сохраненному файлу
        """
        # Преобразуем кортежи в списки для JSON сериализации
        serializable_contexts = [list(context) for context in expected_contexts]
        
        data = {
            "generated_at": datetime.now().isoformat(),
            "test_queries": test_queries,
            "expected_contexts": serializable_contexts,
            "golden_answers": golden_answers
        }
        
        # Создаем имя файла с текущей датой и временем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_data_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return filepath
    
    def generate_test_data(self, num_samples: int = 5, save_to_file: bool = True) -> Tuple[List[str], List[Tuple[str]], List[str], str]:
        """
        Генерирует тестовые данные для оценки RAG-системы
        
        Args:
            num_samples: количество тестовых примеров (по умолчанию 5)
            save_to_file: сохранять ли данные в файл
            
        Returns:
            Кортеж из четырех элементов:
            - test_queries: список тестовых запросов
            - expected_contexts: список кортежей с эталонными контекстами
            - golden_answers: список эталонных ответов
            - filepath: путь к сохраненному файлу (если save_to_file=True)
        """
        test_queries = []
        expected_contexts = []
        golden_answers = []
        
        logger.info(f"Начинаем генерацию {num_samples} тестовых примеров...")
        
        for i in range(num_samples):
            # Извлекаем факты (файлы)
            facts = self.extract_facts()
            
            # Генерируем запрос
            query = self.generate_query(facts)
            
            # Генерируем эталонный ответ
            golden_answer = self.generate_golden_answer(facts, query)
            
            # Добавляем данные в списки
            test_queries.append(query)
            expected_contexts.append(tuple(facts))  # Сохраняем факты как кортеж
            golden_answers.append(golden_answer)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Сгенерировано {i + 1} из {num_samples} примеров")
        
        logger.info("Генерация тестовых данных завершена")
        
        filepath = None
        if save_to_file:
            filepath = self.save_test_data(test_queries, expected_contexts, golden_answers)
        
        return test_queries, expected_contexts, golden_answers, filepath

# Пример использования
if __name__ == "__main__":
    generator = DataGenerator()
    test_queries, expected_contexts, golden_answers, filepath = generator.generate_test_data()
    
    print(f"Тестовые данные сохранены в файл: {filepath}")
    print("\nСгенерированные тестовые данные:")
    for i, (query, context, answer) in enumerate(zip(test_queries, expected_contexts, golden_answers), 1):
        print(f"\nПример {i}:")
        print(f"Запрос: {query}")
        print(f"Контекст: {context}")
        print(f"Эталонный ответ: {answer}") 