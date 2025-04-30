from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.spatial.distance import cdist
from datetime import datetime
from yandex_cloud_ml_sdk import YCloudML
from assistant import Assistant
from utils import get_environment_variables

class RAGEvaluator:
    """Класс для автоматической оценки RAG-ассистента с использованием Yandex Cloud ML"""
    
    def __init__(self, folder_id: str, api_key: str):
        """
        Инициализация оценщика
        
        Args:
            folder_id: ID каталога в Yandex Cloud
            api_key: API ключ для Yandex Cloud
        """
        self.sdk = YCloudML(folder_id=folder_id, auth=api_key)
        self.embedding_model = self.sdk.models.text_embeddings('doc')
        
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Вычисление косинусной схожести между двумя текстами"""
        emb1 = np.array(self.embedding_model.run(text1))
        emb2 = np.array(self.embedding_model.run(text2))
        return 1 - cdist([emb1], [emb2], metric='cosine')[0][0]
    
    def evaluate_context_recall(self, query: str, response: str, reference_contexts: Tuple[str]) -> float:
        """
        Оценка полноты извлечения контекста (Context Recall)
        
        Args:
            query: исходный запрос
            response: ответ системы
            reference_contexts: кортеж эталонных контекстов
            
        Returns:
            Оценка от 0 до 1
        """
        # Вычисляем схожесть с каждым эталонным контекстом
        similarities = [self.calculate_similarity(context, response) for context in reference_contexts]
        # Возвращаем максимальную схожесть
        return max(similarities)
    
    def evaluate_faithfulness(self, response: str, retrieved_context: str) -> float:
        """
        Оценка соответствия ответа извлеченному контексту (Faithfulness)
        
        Args:
            response: ответ системы
            retrieved_context: извлеченный контекст
            
        Returns:
            Оценка от 0 до 1
        """
        return self.calculate_similarity(retrieved_context, response)
    
    def evaluate_factual_correctness(self, response: str, golden_answer: str) -> float:
        """
        Оценка фактической точности ответа (Factual Correctness)
        
        Args:
            response: ответ системы
            golden_answer: эталонный ответ
            
        Returns:
            Оценка от 0 до 1
        """
        return self.calculate_similarity(golden_answer, response)
    
    def evaluate_rag_system(
        self,
        assistant: Assistant,
        test_queries: List[str],
        expected_contexts: List[Tuple[str]],
        golden_answers: List[str]
    ) -> Dict[str, float]:
        """
        Полная оценка RAG-системы с учетом всех извлеченных документов
        
        Args:
            assistant: экземпляр ассистента
            test_queries: список тестовых запросов
            expected_contexts: список кортежей с эталонными контекстами
            golden_answers: список эталонных ответов
            
        Returns:
            Словарь с усредненными оценками:
            - context_recall: полнота извлечения (по всем документам)
            - faithfulness: соответствие ответа контексту (лучший документ)
            - factual_correctness: соответствие ответа эталону
        """
        context_recall_scores = []
        faithfulness_scores = []
        factual_correctness_scores = []
        
        for query, ref_contexts, golden_answer in zip(test_queries, expected_contexts, golden_answers):
            # Создаем новый чат для каждого запроса
            chat = assistant.create_chat()
            
            try:
                # 1. Получаем ответ и извлеченные документы
                response = chat.ask(query)
                retrieved_docs = chat.get_retrieved_context()
                
                # 2. Context Recall: сравниваем с каждым эталонным контекстом
                context_recall = self.evaluate_context_recall(query, response, ref_contexts)
                context_recall_scores.append(context_recall)
                
                # 3. Faithfulness: находим документ, наиболее похожий на ответ
                if retrieved_docs:
                    doc_faith_scores = [
                        self.calculate_similarity(response, doc) 
                        for doc in retrieved_docs
                    ]
                    faithfulness_scores.append(max(doc_faith_scores))
                else:
                    faithfulness_scores.append(0.0)
                
                # 4. Factual Correctness
                fc = self.calculate_similarity(response, golden_answer)
                factual_correctness_scores.append(fc)
            finally:
                # Закрываем чат
                chat.close()
        
        return {
            "context_recall": np.mean(context_recall_scores),
            "faithfulness": np.mean(faithfulness_scores),
            "factual_correctness": np.mean(factual_correctness_scores),
            "coverage": np.mean([min(1, len(retrieved_docs)/5) for _ in test_queries])  # Доп. метрика
        }

# Пример использования
if __name__ == "__main__":
    import json
    import os
    import argparse
    
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Оценка RAG-системы')
    parser.add_argument('--test-data', type=str, help='Путь к файлу с тестовыми данными')
    args = parser.parse_args()
    
    # Конфигурация
    FOLDER_ID, API_KEY = get_environment_variables()
    
    # Загрузка тестовых данных
    if args.test_data:
        if not os.path.exists(args.test_data):
            raise FileNotFoundError(f"Файл с тестовыми данными не найден: {args.test_data}")
        
        with open(args.test_data, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            
        TEST_QUERIES = test_data['test_queries']
        EXPECTED_CONTEXTS = test_data['expected_contexts']
        GOLDEN_ANSWERS = test_data['golden_answers']
        
        print(f"Загружены тестовые данные из файла: {args.test_data}")
        print(f"Дата генерации: {test_data['generated_at']}")
    else:
        # Если файл не указан, генерируем новые данные
        from data_generator import DataGenerator
        generator = DataGenerator()
        TEST_QUERIES, EXPECTED_CONTEXTS, GOLDEN_ANSWERS, filepath = generator.generate_test_data(num_samples=2)
        print(f"Сгенерированы новые тестовые данные и сохранены в: {filepath}")
    
    # Инициализация оценщика
    evaluator = RAGEvaluator(FOLDER_ID, API_KEY)
    
    # Инициализация ассистента
    assistant = Assistant(
        folder_id=FOLDER_ID,
        api_key=API_KEY,
        data_dir="data",
        log_dir="logs",
        model_name="yandexgpt",
        model_version="rc",
        ttl_days=1
    )
    
    # Запуск оценки
    results = evaluator.evaluate_rag_system(
        assistant,
        TEST_QUERIES,
        EXPECTED_CONTEXTS,
        GOLDEN_ANSWERS
    )
    
    print("\nРезультаты оценки:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")