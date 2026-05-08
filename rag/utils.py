import os
import sys
from typing import Optional
try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*args, **kwargs):
        return False
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


def get_environment_variables() -> tuple[Optional[str], Optional[str]]:
    """Load and validate required environment variables.
    
    Returns:
        Tuple of (folder_id, api_key)
        
    Raises:
        SystemExit: If required environment variables are not set
    """
    load_dotenv()
    
    folder_id = os.getenv("YC_FOLDER_ID") or os.getenv("folder_id")
    api_key = os.getenv("YC_API_KEY") or os.getenv("api_key")
    
    if not folder_id or not api_key:
        print("Error: Required environment variables are not set.")
        print("Please make sure .env file exists with the following variables:")
        print("YC_FOLDER_ID=your_folder_id")
        print("YC_API_KEY=your_api_key")
        sys.exit(1)
    return folder_id, api_key
        

def make_ai_tool(model, system_prompt, postprocessing=lambda x: x):
    system = SystemMessage(content=system_prompt)

    def process_message(message: str):
        human = HumanMessage(content=message)
        pseudo_chat = [system, human]
        try:
            result = model.invoke(pseudo_chat).content
        except Exception as e:
            # Логгировать ошибку
            return ''
        return postprocessing(result)

    return process_message
