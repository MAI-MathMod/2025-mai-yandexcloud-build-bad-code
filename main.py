import sys
from assistant import Assistant
from utils import get_environment_variables


def main() -> None:
    """Main function to run the MAI admissions assistant."""
    try:
        folder_id, api_key = get_environment_variables()
        
        print("-" * 55)
        print("| Добро пожаловать в ассистента по поступлению в МАИ! |")
        print("| Введите 'exit' или 'quit', чтобы закончить диалог.  |")
        print("-" * 55)
        
        print('Идёт инициализация ассистента, пожалуйста, подождите...', end='\r', flush=True)
        assistant = Assistant(
            folder_id=folder_id,
            api_key=api_key,
            log_dir="logs"  # Logs will be stored in the 'logs' directory
        )
        print(' ' * 55, end='\r')
        
        # Create a new chat session
        with assistant.create_chat() as chat:
            while True:
                try:
                    message = input("You: ").strip()
                    
                    if message.lower() in ("exit", "quit"):
                        print("Goodbye!")
                        break
                        
                    if not message:
                        print("Please enter a message.")
                        continue
                        
                    response = chat.ask(message)
                    print(f"Assistant: {response}")
                    
                except KeyboardInterrupt:
                    print("\nGoodbye!")
                    break
                    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()