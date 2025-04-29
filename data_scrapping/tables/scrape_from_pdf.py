import os
import camelot
import argparse
from pathlib import Path


def scrape_tables(pdf_path, output_dir):
    """
    Извлечение таблиц из PDF файла и сохранение их в указанную директорию

    Args:
        pdf_path (str): путь к PDF файлу
        output_dir (str): директория для сохранения результатов
    """
    # Проверяем существование файла
    if not os.path.exists(pdf_path):
        print(f"Ошибка: Файл {pdf_path} не найден")
        return

    # Создаем директорию для сохранения результатов, если она не существует
    os.makedirs(output_dir, exist_ok=True)

    # Получаем имя файла без расширения для формирования имен выходных файлов
    pdf_filename = Path(pdf_path).stem

    try:
        # Читаем таблицы из PDF
        print(f"Извлечение таблиц из {pdf_path}...")
        tables = camelot.read_pdf(pdf_path, pages='all')

        print(f"Найдено {len(tables)} таблиц")

        # Сохраняем каждую таблицу в CSV формате
        for i, table in enumerate(tables):
            output_path = os.path.join(output_dir, f"{pdf_filename}_table_{i+1}.csv")
            table.to_csv(output_path)
            print(f"Таблица {i+1} сохранена в {output_path}")


        # Сохраняем сводную информацию
        # if len(tables) > 0:
        #     summary_path = os.path.join(output_dir, f"{pdf_filename}_summary.html")
        #     tables.export(summary_path, f='html')
        #     print(f"Сводная информация сохранена в {summary_path}")

    except Exception as e:
        print(f"Ошибка при обработке PDF: {e}")


if __name__ == "__main__":
    # Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description='Извлечение таблиц из PDF файла')
    parser.add_argument('pdf_path', type=str, help='Путь к PDF файлу')
    parser.add_argument('--output_dir', type=str, default='extracted_tables',
                        help='Директория для сохранения извлеченных таблиц (по умолчанию: extracted_tables)')

    args = parser.parse_args()
    scrape_tables(args.pdf_path, args.output_dir)


