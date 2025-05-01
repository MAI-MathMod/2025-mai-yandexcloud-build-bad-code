import sys
import boto3
from assistant import Assistant
from utils import get_environment_variables
import os
from typing import Dict
from datetime import datetime
from yandex_cloud_ml_sdk import YCloudML

def main() -> None:
    """Главная функция ассистента приёмной комиссии МАИ"""
    try:
        # Получение переменных окружения
        folder_id, api_key = get_environment_variables()
        
        # Инициализация клиента YMQ
        yc_access_id = os.getenv('YC_SERVICE_ACCESS_ID')
        yc_access_key = os.getenv('YC_ACCESS_KEY')
        
        if not all([yc_access_id, yc_access_key]):
            raise ValueError("Не найдены учетные данные YMQ в переменных окружения")
            
        session = boto3.session.Session(
            aws_access_key_id=yc_access_id,
            aws_secret_access_key=yc_access_key,
            region_name='ru-central1'
        )
        
        sqs = session.client(
            service_name='sqs',
            endpoint_url='https://message-queue.api.cloud.yandex.net'
        )
        
        # URL очередей из переменных окружения
        bot_events_queue_url = os.getenv('BOT_EVENTS_QUEUE_URL')
        rag_response_queue_url = os.getenv('RAG_RESPONSE_QUEUE_URL')
        
        if not bot_events_queue_url or not rag_response_queue_url:
            raise ValueError("URL очередей не найдены в переменных окружения")
        
        # Инициализация ассистента
        print("\n🔄 Инициализация AI-ассистента МАИ...", end='\r', flush=True)
        sdk = YCloudML(folder_id=folder_id, auth=api_key)
        assistant = Assistant(sdk)
        print(" " * 50, end='\r')
        
        # Словарь активных чатов
        active_chats: set = set()
        
        print("\n✅ Ассистент готов к работе. Ожидание сообщений...\n")
        
        # Основной цикл обработки сообщений
        while True:
            try:
                # Получение сообщения из очереди
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
                message_attrs = message.get('MessageAttributes', {})

                # Извлечение информации о пользователе
                username = next((
                    message_attrs[key]['StringValue'] 
                    for key in ['Username', 'username', 'UserName', 'user_name', 'FromUser'] 
                    if key in message_attrs and 'StringValue' in message_attrs[key]
                ), 'Анонимный пользователь')

                chat_id = None
                for key in ['ChatID', 'chat_id', 'ChatId']:
                    if key in message_attrs:
                        try:
                            chat_id = int(message_attrs[key]['StringValue'])
                            break
                        except (ValueError, KeyError):
                            continue

                if not chat_id:
                    print(f"⚠️ [{datetime.now().strftime('%H:%M:%S')}] Сообщение от {username}: отсутствует ChatID")
                    continue

                # Логирование нового пользователя
                if chat_id not in active_chats:
                    active_chats.add(assistant.create_chat(chat_id))
                    print(f"\n🌟 [{datetime.now().strftime('%H:%M:%S')}] Новый пользователь: {username} (ID: {chat_id})")
                
                # Обработка запроса
                print(f"\n💬 [{datetime.now().strftime('%H:%M:%S')}] Запрос от {username}: {user_message[:50]}...")
                
                assistant_response = assistant.ask(chat_id, user_message)
                print(f"📨 [{datetime.now().strftime('%H:%M:%S')}] Ответ для {username}: {assistant_response[:50]}...")
                
                # Отправка ответа
                sqs.send_message(
                    QueueUrl=rag_response_queue_url,
                    MessageBody=assistant_response,
                    MessageGroupId=str(chat_id),
                    MessageAttributes={
                        'ChatID': {
                            'StringValue': str(chat_id),
                            'DataType': 'Number'
                        },
                        'Username': {
                            'StringValue': username,
                            'DataType': 'String'
                        }
                    }
                )
                
                # Удаление обработанного сообщения
                sqs.delete_message(
                    QueueUrl=bot_events_queue_url,
                    ReceiptHandle=receipt_handle
                )

            except KeyboardInterrupt:
                print("\n🛑 Завершение работы...")
                # Закрытие всех чатов
                # for cid in active_chats:
                #     chat.close()
                break
            
            except Exception as e:
                error_time = datetime.now().strftime('%H:%M:%S')
                username_info = f" ({username})" if 'username' in locals() else ""
                print(f"\n⚠️ [{error_time}] Ошибка обработки{username_info}: {str(e)}")
                continue

    except Exception as e:
        print(f"\n⛔ Критическая ошибка: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()