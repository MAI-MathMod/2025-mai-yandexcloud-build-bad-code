from typing import List, Dict, Any
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
    
    def evaluate_context_recall(self, query: str, response: str, reference_context: str) -> float:
        """
        Оценка полноты извлечения контекста (Context Recall)
        
        Args:
            query: исходный запрос
            response: ответ системы
            reference_context: эталонный контекст
            
        Returns:
            Оценка от 0 до 1
        """
        return self.calculate_similarity(reference_context, response)
    
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
        assistant: Any,
        test_queries: List[str],
        expected_contexts: List[str],
        golden_answers: List[str]
    ) -> Dict[str, float]:
        """
        Полная оценка RAG-системы с учетом всех извлеченных документов
        
        Args:
            assistant: экземпляр ассистента
            test_queries: список тестовых запросов
            expected_contexts: список эталонных контекстов
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
        
        with assistant:
            for query, ref_context, golden_answer in zip(test_queries, expected_contexts, golden_answers):
                # 1. Получаем ответ и извлеченные документы
                response = assistant.ask(query)
                retrieved_docs = assistant.get_retrieved_context()
                
                # 2. Context Recall: сравниваем КАЖДЫЙ документ с эталоном
                doc_recall_scores = [
                    self.calculate_similarity(ref_context, doc) 
                    for doc in retrieved_docs
                ]
                context_recall_scores.append(np.max(doc_recall_scores))  # Берем лучший
                
                # 3. Faithfulness: находим документ, наиболее похожий на ответ
                if retrieved_docs:
                    doc_faith_scores = [
                        self.calculate_similarity(response, doc) 
                        for doc in retrieved_docs
                    ]
                    faithfulness_scores.append(np.max(doc_faith_scores))
                else:
                    faithfulness_scores.append(0.0)
                
                # 4. Factual Correctness (не зависит от документов)
                fc = self.calculate_similarity(response, golden_answer)
                factual_correctness_scores.append(fc)
        
        return {
            "context_recall": np.mean(context_recall_scores),
            "faithfulness": np.mean(faithfulness_scores),
            "factual_correctness": np.mean(factual_correctness_scores),
            "coverage": np.mean([min(1, len(retrieved_docs)/5) for _ in test_queries])  # Доп. метрика
        }

# Пример использования
if __name__ == "__main__":
    # Конфигурация
    FOLDER_ID, API_KEY = get_environment_variables()
    
    # Тестовые данные
    TEST_QUERIES = [
        "Какие документы нужны для поступления в МАИ?",
        "Когда начинается приемная кампания?",
        "Какие направления подготовки есть в МАИ?"
    ]
    
    EXPECTED_CONTEXTS = [
        "Для поступления в МАИ необходимы: паспорт, документ об образовании, результаты ЕГЭ, 2 фотографии 3x4.",
        "Приемная кампания в МАИ традиционно начинается 20 июня и заканчивается 26 июля.",
        "МАИ предлагает направления: Авиастроение, Ракетные комплексы, Информатика, Радиоэлектроника."
    ]
    
    GOLDEN_ANSWERS = [
        "Для поступления в МАИ вам понадобятся: паспорт, аттестат или диплом, результаты ЕГЭ и 2 фотографии 3x4 см.",
        "Прием документов в МАИ начинается 20 июня каждого года.",
        "В МАИ доступны такие направления как Авиастроение, Ракетные комплексы, Информатика и другие."
    ]
    
    # Инициализация оценщика
    evaluator = RAGEvaluator(FOLDER_ID, API_KEY)
    
    # Инициализация ассистента (ваш код)
    assistant = Assistant(
        folder_id=FOLDER_ID,
        api_key=API_KEY,
        data_dir="data",
        log_dir="logs"
    )
    
    # Запуск оценки
    results = evaluator.evaluate_rag_system(
        assistant,
        TEST_QUERIES,
        EXPECTED_CONTEXTS,
        GOLDEN_ANSWERS
    )
    
    print("Результаты оценки:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")