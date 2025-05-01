from langchain_chroma import Chroma  # Изменённый импорт
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional, Dict
from yandex_cloud_ml_sdk import YCloudML
import os
import glob
from bs4 import BeautifulSoup
from utils import get_environment_variables

class SearchEngine:
    def __init__(
        self,
        data_folder: str,
        sdk: YCloudML,
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.data_folder = data_folder
        self.sdk = sdk
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorstore: Optional[Chroma] = None
        self.retriever = None
        
        # Инициализация эмбеддингов
        self.doc_embeddings = YandexAsymmetricEmbeddings(sdk, mode="doc")
        self.query_embeddings = YandexAsymmetricEmbeddings(sdk, mode="query")
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def _load_text_file(self, file_path: str) -> str:
        """Загружает содержимое текстового файла"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def _load_documents(self) -> List[str]:
        """Рекурсивно загружает документы из различных форматов файлов"""
        documents = []
        
        # Поддерживаемые форматы файлов
        extensions = ['*.txt', '*.csv', '*.md']
        
        for ext in extensions:
            for file_path in glob.glob(os.path.join(self.data_folder, '**', ext), recursive=True):
                try:
                    content = self._load_text_file(file_path)
                    documents.append(Document(content))
                except Exception as e:
                    print(f"Ошибка при обработке файла {file_path}: {e}")
        
        # Разбиваем документы на чанки
        if documents:
            return self.text_splitter.split_documents(documents)
        return []
    
    def _initialize_vectorstore(self, documents: List[Document]):
        """Инициализирует Chroma с документами"""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.doc_embeddings,
            persist_directory=self.persist_directory,
            metadatas=metadatas
        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 1},
            embedding_function=self.query_embeddings.embed_query  # Переопределяем для поиска
        )
    
    def _load_existing_vectorstore(self):
        """Загружает существующую базу Chroma"""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.doc_embeddings
        )

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 1},
            embedding_function=self.query_embeddings.embed_query  # Переопределяем для поиска
        )
    
    def initialize(self):
        """Инициализирует или загружает базу документов"""
        if not os.path.exists(self.persist_directory) or not os.listdir(self.persist_directory):
            documents = self._load_documents()
            if documents:
                self._initialize_vectorstore(documents)
            else:
                raise ValueError("Не найдено документов для индексации")
        else:
            self._load_existing_vectorstore()
    
    def search(self, query: str, n_results: int = 5) -> List[Document]:
        """
        Выполняет поиск по документам
        
        :param query: Поисковый запрос
        :return: Список релевантных документов
        """

        if self.retriever is None:
            self.initialize()
        self.retriever.search_kwargs['k'] = n_results
        return [d.page_content for d in self.retriever.invoke(query)]  # Используем invoke вместо get_relevant_documents

class YandexAsymmetricEmbeddings(Embeddings):
    def __init__(self, sdk: YCloudML, mode: str = "doc"):
        self.sdk = sdk
        self.mode = mode  # "doc" или "query"
        self.model = sdk.models.text_embeddings(mode)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Для документов (режим 'doc')"""
        return [list(self.model.run(text)) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Для запросов (режим 'query')"""
        return list(self.model.run(text))

# Пример использования
if __name__ == "__main__":
    folder_id, api_key = get_environment_variables()
    
    # Инициализация Yandex Cloud ML
    sdk = YCloudML(folder_id=folder_id, auth=api_key)
    sdk.setup_default_logging()
    
    # Создаем и инициализируем поисковый движок
    search_engine = SearchEngine(
        data_folder="./data",  # Папка с документами
        sdk=sdk,
        persist_directory="./chroma_db",
        chunk_size=800,  # Можно настроить под ваши нужды
        chunk_overlap=100
    )
    search_engine.initialize()
    
    # Пример поиска
    results = search_engine.search("Какие проходные баллы были на прикладную математику и информатику в 2024 году?", n_results=1)
    for doc in results:
        print('-' * 50)
        print(doc)











