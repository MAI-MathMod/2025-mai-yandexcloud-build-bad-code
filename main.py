import sys
import boto3
from assistant import Assistant
from utils import get_environment_variables
import os
from typing import Dict

def main() -> None:
    """Main function to run the MAI admissions assistant."""
    try:
        folder_id, api_key = get_environment_variables()
        
        yc_access_id = os.getenv('YC_SERVICE_ACCESS_ID')
        yc_access_key = os.getenv('YC_ACCESS_KEY')
        
        if not all([yc_access_id, yc_access_key]):
            raise ValueError("YMQ credentials not found in environment variables")
            
        session = boto3.session.Session(
            aws_access_key_id=yc_access_id,
            aws_secret_access_key=yc_access_key,
            region_name='ru-central1'
        )
        
        sqs = session.client(
            service_name='sqs',
            endpoint_url='https://message-queue.api.cloud.yandex.net'
        )
        
        bot_events_queue_url = os.getenv('BOT_EVENTS_QUEUE_URL')
        rag_response_queue_url = os.getenv('RAG_RESPONSE_QUEUE_URL')
        
        if not bot_events_queue_url or not rag_response_queue_url:
            raise ValueError("Queue URLs not found in environment variables")
        
        print('Инициализация ассистента...', end='\r', flush=True)
        assistant = Assistant(
            folder_id=folder_id,
            api_key=api_key,
            log_dir="logs"
        )
        print(' ' * 30, end='\r')
        
        active_chats: Dict[int, Assistant.Chat] = {}
        
        while True:
            try:
                # Получение сообщения из очереди с атрибутами
                response = sqs.receive_message(
                    QueueUrl=bot_events_queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=20,
                    MessageAttributeNames=['All']
                )
                
                if 'Messages' not in response:
                    continue
                
                message = response['Messages'][0]
                receipt_handle = message['ReceiptHandle']
                user_message = message['Body']
                
                # Извлечение ChatID из атрибутов сообщения
                message_attrs = message.get('MessageAttributes', {})
                chat_id_attr = message_attrs.get('ChatID', {}).get('StringValue')
                
                if not chat_id_attr:
                    print("ChatID attribute missing in message")
                    continue
                    
                try:
                    chat_id = int(chat_id_attr)
                except ValueError:
                    print(f"Invalid ChatID format: {chat_id_attr}")
                    continue
                
                # Создание или получение существующего чата
                if chat_id not in active_chats:
                    active_chats[chat_id] = assistant.create_chat()
                    print(f"Создан новый чат для ChatID: {chat_id}")
                
                chat = active_chats[chat_id]
                
                # Генерация ответа
                assistant_response = chat.ask(user_message)
                
                # Отправка ответа в очередь с корректными атрибутами
                sqs.send_message(
                    QueueUrl=rag_response_queue_url,
                    MessageBody=assistant_response,
                    MessageGroupId=str(chat_id),
                    MessageAttributes={
                        'ChatID': {
                            'StringValue': str(chat_id),
                            'DataType': 'Number'
                        }
                    }
                )
                
                # Удаление обработанного сообщения
                sqs.delete_message(
                    QueueUrl=bot_events_queue_url,
                    ReceiptHandle=receipt_handle
                )
                
            except KeyboardInterrupt:
                print("\nРабота завершена.")
                # Закрытие всех активных чатов перед выходом
                for chat in active_chats.values():
                    chat.close()
                break
            except Exception as e:
                print(f"Ошибка обработки сообщения: {str(e)}")
                continue

    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()