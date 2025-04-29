import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import argparse
from typing import Optional, List, Dict, Any


def get_html_content(url: str) -> Optional[str]:
    """
    Получает HTML-контент страницы по указанному URL.
    
    Args:
        url: URL-адрес для загрузки
        
    Returns:
        Строка с HTML-контентом или None в случае ошибки
    """
    try:
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        
        if response.status_code != 200:
            print(f"Ошибка при загрузке страницы: {response.status_code}")
            return None
            
        return response.text
    except requests.RequestException as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return None


def extract_table_data(table) -> List[List[str]]:
    """
    Извлекает данные из HTML-таблицы.
    
    Args:
        table: HTML-элемент таблицы
        
    Returns:
        Список строк таблицы, каждая из которых представлена списком ячеек
    """
    rows = []
    for row in table.find_all('tr'):
        cells = row.find_all(['td', 'th'])
        if cells:  # Пропускаем пустые строки
            row_data = [cell.get_text(strip=True) for cell in cells]
            rows.append(row_data)
    
    return rows


def save_table_to_csv(data: List[List[str]], filename: str) -> bool:
    """
    Сохраняет данные таблицы в CSV-файл.
    
    Args:
        data: Списки данных таблицы
        filename: Имя файла для сохранения
        
    Returns:
        True если таблица была успешно сохранена, иначе False
    """
    if not data:
        return False
        
    # Создаем DataFrame из данных
    df = pd.DataFrame(data)
    
    # Устанавливаем первую строку как заголовок, если таблица не пустая
    if df.shape[0] > 0:
        headers = df.iloc[0]
        df.columns = headers
        df = df.iloc[1:].reset_index(drop=True)
    
    try:
        # Сохраняем таблицу в CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        return True
    except Exception as e:
        print(f"Ошибка при сохранении таблицы в {filename}: {e}")
        return False


def scrape_tables(url: str, output_dir: str) -> Optional[int]:
    """
    Извлекает все таблицы с веб-страницы и сохраняет их в CSV-файлы.
    
    Args:
        url: URL веб-страницы для извлечения таблиц
        output_dir: Директория для сохранения файлов
        
    Returns:
        Количество извлеченных таблиц или None в случае ошибки
    """
    # Получаем HTML страницы
    html_content = get_html_content(url)
    if not html_content:
        return None

    # Создаем объект BeautifulSoup для парсинга HTML
    soup = BeautifulSoup(html_content, 'html.parser')

    # Находим все таблицы на странице
    tables = soup.find_all('table')
    
    if not tables:
        print("На странице не найдено таблиц")
        return 0
    
    print(f"Найдено таблиц: {len(tables)}")
    
    # Создаем директорию для сохранения файлов, если её нет
    os.makedirs(output_dir, exist_ok=True)
    
    # Счетчик успешно сохраненных таблиц
    saved_tables_count = 0
    
    # Обрабатываем каждую таблицу
    for i, table in enumerate(tables):
        # Извлекаем данные из таблицы
        table_data = extract_table_data(table)
        
        if not table_data:
            print(f"Таблица {i+1} не содержит данных, пропускаем")
            continue
        
        # Формируем имя файла
        filename = os.path.join(output_dir, f"table_{i+1}.csv")
        
        # Сохраняем таблицу
        if save_table_to_csv(table_data, filename):
            saved_tables_count += 1
            print(f"Таблица {i+1} сохранена в {filename}")
    
    return saved_tables_count


def main():
    """Основная функция для запуска скрипта из командной строки"""
    parser = argparse.ArgumentParser(description='Извлечение таблиц с веб-страницы')
    parser.add_argument('url', type=str, help='URL веб-страницы для извлечения таблиц')
    parser.add_argument('--output_dir', type=str, default='extracted_tables',
                       help='Директория для сохранения извлеченных таблиц (по умолчанию: extracted_tables)')
    
    args = parser.parse_args()
    
    result = scrape_tables(args.url, args.output_dir)
    
    if result is None:
        print("Не удалось извлечь таблицы из-за ошибки.")
    elif result == 0:
        print("Таблицы не были найдены или не содержали данных.")
    else:
        print(f"Успешно сохранено таблиц: {result}")


if __name__ == "__main__":
    main()