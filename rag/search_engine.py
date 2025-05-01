from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Optional, Dict
from yandex_cloud_ml_sdk import YCloudML
import os
import glob
from tqdm import tqdm
from bs4 import BeautifulSoup
from utils import get_environment_variables
from langchain_community.retrievers import BM25Retriever
from langchain.vectorstores.base import VectorStoreRetriever
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

class AsymmetricRetriever(VectorStoreRetriever):
    query_embedding_function: callable

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_vector = self.query_embedding_function(query)
        return self.vectorstore.similarity_search_by_vector(query_vector, k=self.search_kwargs.get("k", 5))

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
    
    def _process_file(self, args):
        """Читает файл и сразу возвращает его чанки."""
        file_path, chunk_size, chunk_overlap = args
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Ошибка чтения {file_path}: {e}")
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        # split_documents ожидает список Document
        docs = splitter.split_documents([Document(page_content=content)])
        return docs

    def _load_documents(self) -> List[Document]:
        """Загружает и сплитит файлы параллельно."""
        extensions = ['*.txt', '*.csv', '*.md', '*.html', '*.json']
        all_files = list(itertools.chain.from_iterable(
            glob.glob(os.path.join(self.data_folder, '**', ext), recursive=True)
            for ext in extensions
        ))

        chunks: List[Document] = []
        total = len(all_files)
        print(f"Найдено файлов: {total}")

        # Подготовим аргументы для каждого процесса
        args_list = [
            (fp, self.chunk_size, self.chunk_overlap)
            for fp in all_files
        ]

        # Пул процессов
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(self._process_file, args): args[0] for args in args_list}
            for i, future in enumerate(as_completed(futures), start=1):
                file_path = futures[future]
                try:
                    docs = future.result()
                    chunks.extend(docs)
                except Exception as e:
                    print(f"Ошибка при обработке {file_path}: {e}")
                print(f'Processed {i}/{total} files', end='\r')

        print(f"\nВсего чанков: {len(chunks)}")
        return chunks
    
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
    
    def _load_existing_vectorstore(self):
        """Загружает существующую базу Chroma"""
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.doc_embeddings
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
        
        # Настройка ретриверов
        data = self.vectorstore.get()
        texts = data['documents']
        
        # Создание BM25 ретривера
        bm25_retriever = BM25Retriever.from_texts(texts)
        bm25_retriever.k = 5
        
        # Создание Chroma ретривера с асимметричными эмбеддингами
        chroma_retriever = AsymmetricRetriever(
            vectorstore=self.vectorstore,
            query_embedding_function=self.query_embeddings.embed_query,
            search_kwargs={"k": 5}
        )
        
        # Создание EnsembleRetriever
        ensemble_retriever = EnsembleRetriever(retrievers=[chroma_retriever, bm25_retriever])
        
        # Установка self.retriever
        self.retriever = ensemble_retriever
    
    def search(self, query: str, n_results: int = 5) -> List[str]:
        """
        Выполняет поиск по документам
        
        :param query: Поисковый запрос
        :return: Список релевантных текстов документов
        """
        if self.retriever is None:
            self.initialize()
        docs = self.retriever.invoke(query)
        return [d.page_content for d in docs[:n_results]]

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
    
    print('searching....')
    # Пример поиска
    results = search_engine.search("Какие проходные баллы были на прикладную математику в 2024 году?", n_results=6)
    for doc in results:
        print('-' * 50)
        print(doc)