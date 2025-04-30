import os
import sys
from typing import Optional
from dotenv import load_dotenv


def get_environment_variables() -> tuple[Optional[str], Optional[str]]:
    """Load and validate required environment variables.
    
    Returns:
        Tuple of (folder_id, api_key)
        
    Raises:
        SystemExit: If required environment variables are not set
    """
    load_dotenv()
    
    folder_id = os.getenv("folder_id")
    api_key = os.getenv("api_key")
    
    if not folder_id or not api_key:
        print("Error: Required environment variables are not set.")
        print("Please make sure .env file exists with the following variables:")
        print("folder_id=your_folder_id")
        print("api_key=your_api_key")
        sys.exit(1)
    return folder_id, api_key
        