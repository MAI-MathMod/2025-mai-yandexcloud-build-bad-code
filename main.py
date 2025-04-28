import os
from assistant import Assistant
from dotenv import load_dotenv
load_dotenv()

folder_id = os.getenv("folder_id")
api_key = os.getenv("api_key")

assistant = Assistant(folder_id, api_key)

assistant.start_chat()
while True:
    message = input('user: ')
    response = assistant.ask(message)
    print(f'assistant: {response}')
assistant.end_chat()