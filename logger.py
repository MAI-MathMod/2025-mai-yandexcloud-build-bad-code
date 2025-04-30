import logging
import json
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path


class AssistantLogger:
    """Custom logger for MAI Assistant with structured logging."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger("mai_assistant")
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        self.json_formatter = JsonFormatter()
        
        # Create handlers
        self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Setup file and console handlers."""
        # File handler for JSON logs
        log_file = self.log_dir / f"assistant_{datetime.now().strftime('%Y%m%d')}.jsonl"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(self.json_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for human-readable logs
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(console_handler)
    
    def log_interaction(
        self,
        message: str,
        response: str,
        start_time: datetime,
        end_time: datetime,
        citations: List[Dict[str, Any]]
    ) -> None:
        """Log an interaction with the assistant.
        
        Args:
            message: User's message
            response: Assistant's response
            start_time: When the message was received
            end_time: When the response was received
            citations: List of citations used in the response
        """
        log_data = {
            "timestamp": start_time.isoformat(),
            "message": message,
            "response": response,
            "processing_time": (end_time - start_time).total_seconds(),
            "citations": [
                {
                    "source": citation.get("source", ""),
                    "content": citation.get("content", "")
                }
                for citation in citations
            ]
        }
        
        self.logger.info("", extra={"log_data": log_data})


class JsonFormatter(logging.Formatter):
    """Custom formatter for JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        if hasattr(record, "log_data"):
            return json.dumps(record.log_data, ensure_ascii=False)
        return json.dumps({
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "message": record.getMessage()
        }, ensure_ascii=False) 