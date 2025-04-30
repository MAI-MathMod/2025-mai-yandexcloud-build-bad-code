from typing import Optional, List, Dict, Any
from datetime import datetime
from yandex_cloud_ml_sdk import YCloudML
from search_system import upload_data, make_search_tool
from logger import AssistantLogger
from usefull_promts import QueryRefiner, ProactiveAssistant, TopicsExtractor, question_topics


class Assistant:
    """AI Assistant resource manager for MAI admissions office.
    
    This class manages the Yandex Cloud ML resources, search capabilities,
    and provides a factory method to create chat sessions.
    """
    
    DEFAULT_INSTRUCTION = (
        "Ты - работник приёмной комиссии Московского Авиационного Института. "
        "Система автоматически напомнит тебе релевантную информацию по вопросу, тебе необходимо "
        "внимательно с ней ознакомиться и ответить на вопрос. Если в предоставленной информации нет "
        "ответа на вопрос, честно скажи пользователю, что информации не найдено, или задай ему уточняющий вопрос."
    )
    
    def __init__(
        self,
        folder_id: str,
        api_key: str,
        data_dir: str = "data",
        instruction: Optional[str] = None,
        model_name: str = "yandexgpt",
        model_version: str = "rc",
        ttl_days: int = 1,
        log_dir: str = "logs"
    ):
        """Initialize the Assistant resource manager.
        
        Args:
            folder_id: Yandex Cloud folder ID
            api_key: Yandex Cloud API key
            data_dir: Directory containing data files for search
            instruction: Custom instruction for the assistant
            model_name: Name of the model to use
            model_version: Version of the model
            ttl_days: Time to live for assistant and thread resources
            log_dir: Directory to store log files
        """
        self.folder_id = folder_id
        self.api_key = api_key
        self.ttl_days = ttl_days
        
        # Initialize logger
        self.logger = AssistantLogger(log_dir)
        
        # Initialize SDK and model
        self.sdk = YCloudML(folder_id=folder_id, auth=api_key)
        self.model = self.sdk.models.completions(model_name, model_version=model_version)
        
        # Initialize search capabilities
        uploaded_files = upload_data(self.sdk, data_dir)
        self.search_tool = make_search_tool(self.sdk, uploaded_files)
        
        # Initialize prompt processors
        self.query_refiner = QueryRefiner()
        self.proactive_assistant = ProactiveAssistant()
        self.topics_extractor = TopicsExtractor(question_topics)
        
        # Initialize assistant
        self.assistant = self._create_assistant(
            instruction=instruction or self.DEFAULT_INSTRUCTION,
            tools=[self.search_tool]
        )
    
    def _create_assistant(self, instruction: str, tools: Optional[List[Any]] = None) -> Any:
        """Create a new assistant instance.
        
        Args:
            instruction: Assistant's instruction
            tools: List of tools available to the assistant
            
        Returns:
            Created assistant instance
        """
        kwargs = {"tools": tools} if tools else {}
        return self.sdk.assistants.create(
            self.model,
            ttl_days=self.ttl_days,
            expiration_policy="since_last_active",
            **kwargs
        ).update(instruction=instruction)
    
    def create_chat(self) -> 'Chat':
        """Create a new chat session.
        
        Returns:
            A new Chat instance
        """
        return Chat(self)


class Chat:
    """Represents a single chat session with the assistant."""
    
    def __init__(self, assistant: Assistant):
        """Initialize a new chat session.
        
        Args:
            assistant: The Assistant instance to use for this chat
        """
        self.assistant = assistant
        self.thread = assistant.sdk.threads.create(
            ttl_days=assistant.ttl_days,
            expiration_policy="static"
        )
        self.last_citations = []
        self.clarification_questions = []
        self.current_question = None
        self.original_question = None
    
    def _process_question(self, message: str) -> str:
        """Process the question through the pipeline of prompt processors.
        
        Args:
            message: Original user message
            
        Returns:
            Final refined question with tags
        """
        # Step 1: Refine the query
        refined_query = self.assistant.query_refiner(message)
        if not refined_query:
            refined_query = message
            
        # Step 2: Get clarification questions
        clarification_questions = self.assistant.proactive_assistant(refined_query)
        if clarification_questions:
            self.clarification_questions = clarification_questions
            self.original_question = refined_query
            return f"Для более точного ответа, пожалуйста, уточните:\n{clarification_questions[0]}"
            
        # Step 3: Extract topics
        topics = self.assistant.topics_extractor(refined_query)
        if topics:
            topics_str = ", ".join(topics)
            return f"{refined_query}\n\nТеги для поиска: {topics_str}"
            
        return refined_query
    
    def ask(self, message: str) -> str:
        """Send a message to the assistant and get a response.
        
        Args:
            message: User's message
            
        Returns:
            Assistant's response
        """
        start_time = datetime.now()
        
        # If we have clarification questions and this is an answer to one
        if self.clarification_questions:
            self.clarification_questions.pop(0)  # Remove the answered question
            if self.clarification_questions:
                # If there are more questions, ask the next one
                return f"Спасибо за уточнение. Пожалуйста, ответьте на следующий вопрос:\n{self.clarification_questions[0]}"
            else:
                # If all questions are answered, process the original question
                message = self.original_question
                self.original_question = None
        
        # Process the question through our pipeline
        processed_message = self._process_question(message)
        
        # If the processed message is a clarification request, return it directly
        if processed_message.startswith("Для более точного ответа"):
            return processed_message
            
        # Otherwise, proceed with the normal assistant flow
        self.thread.write(processed_message)
        run = self.assistant.assistant.run(self.thread)
        result = run.wait()
        end_time = datetime.now()
        
        # Extract citations and log the interaction
        citations = self._extract_citations(result)
        self.assistant.logger.log_interaction(
            message=message,
            response=result.text,
            start_time=start_time,
            end_time=end_time,
            citations=citations
        )
        self.last_citations = citations
        
        return result.text
    
    def close(self) -> None:
        """Close the chat session."""
        if self.thread:
            self.thread.delete()
            self.thread = None
    
    def __enter__(self):
        """Context manager entry point."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.close()

    def _extract_citations(self, run_result: Any) -> List[Dict[str, Any]]:
        """Extract citations from the run result.
        
        Args:
            run_result: Result from assistant.run()
            
        Returns:
            List of citations with source and content
        """
        citations = []
        for citation in run_result.citations:
            for source in citation.sources:
                if source.type != "filechunk":
                    continue
                citations.append({
                    "source": source.parts[0]
                })
        return citations

    def get_retrieved_context(self) -> List[str]:
        """Возвращает последний извлеченный контекст"""
        if self.last_citations:
            return [c['source'] for c in self.last_citations]
        return [""]
