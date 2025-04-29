import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

def scrape_tables():
    # Получаем HTML страницы
    response = requests.get(url)
    response.encoding = 'utf-8'  # Устанавливаем кодировку

    if response.status_code != 200:
        print(f"Ошибка при загрузке страницы: {response.status_code}")
        return None

    # Создаем объект BeautifulSoup для парсинга HTML
    soup = BeautifulSoup(response.text, 'html.parser')

    # Находим все таблицы на странице
    tables = soup.find_all('table')
    
    if not tables:
        print("На странице не найдено таблиц")
        return None
    
    print(f"Найдено таблиц: {len(tables)}")
    
    # Создаем директорию для сохранения файлов, если её нет
    output_dir = "mai_tables"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Обрабатываем каждую таблицу
    for i, table in enumerate(tables):
        # Извлекаем данные из таблицы
        rows = []
        for row in table.find_all('tr'):
            cells = row.find_all(['td', 'th'])
            if cells:  # Пропускаем пустые строки
                row_data = [cell.get_text(strip=True) for cell in cells]
                rows.append(row_data)
        
        if not rows:
            continue  # Пропускаем пустые таблицы
        
        # Создаем DataFrame из данных
        df = pd.DataFrame(rows)
        
        # Устанавливаем первую строку как заголовок, если таблица не пустая
        if df.shape[0] > 0:
            headers = df.iloc[0]
            df.columns = headers
            df = df.iloc[1:].reset_index(drop=True)
        
        # Формируем имя файла
        filename = f"{output_dir}/table_{i+1}.csv"
        
        # Сохраняем таблицу в CSV
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Таблица {i+1} сохранена в {filename}")
    
    return True

if __name__ == "__main__":
    result = scrape_tables()
    if result:
        print("Все таблицы успешно сохранены.")