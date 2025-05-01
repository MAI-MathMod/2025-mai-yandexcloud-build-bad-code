import requests
from bs4 import BeautifulSoup, Tag
from typing import Dict, List, Optional
import os
from urllib.parse import urlparse


spec_urls = [
    'https://priem.mai.ru/spec/',
    'https://priem.mai.ru/spec/apply/',
    'https://priem.mai.ru/spec/tuition-fees/',
    'https://priem.mai.ru/spec/achievements/',
    'https://priem.mai.ru/spec/offers/',
    'https://priem.mai.ru/spec/tests/',
    'https://priem.mai.ru/spec/hostel/'
]

base_urls = [
    'https://priem.mai.ru/base/',
    'https://priem.mai.ru/base/apply/',
    'https://priem.mai.ru/base/tuition-fees/',
    'https://priem.mai.ru/base/achievements/',
    'https://priem.mai.ru/base/priveleges/',
    'https://priem.mai.ru/base/offers/',
    # 'https://priem.mai.ru/base/score/',
    'https://priem.mai.ru/base/tests/',
    'https://priem.mai.ru/base/hostel/'
]


def html_to_markdown(element: Tag) -> str:
    """
    Преобразует HTML-элемент (или дерево) в markdown-строку с поддержкой заголовков, списков и таблиц.
    Исправляет склеивание слов при наличии inline-тегов.
    """
    lines = []
    def process(el, prefix=""):
        if isinstance(el, str):
            lines.append(prefix + el.strip())
            return
        if el.name in [f'h{i}' for i in range(1, 7)]:
            level = int(el.name[1])
            lines.append(f"{'#' * level} {el.get_text(strip=True)}\n")
        elif el.name in ['ul', 'ol']:
            for li in el.find_all('li', recursive=False):
                lines.append(f"- {' '.join(li.stripped_strings)}\n")
        elif el.name == 'table':
            rows = el.find_all('tr')
            table = []
            for row in rows:
                cols = [col.get_text(strip=True) for col in row.find_all(['td', 'th'])]
                table.append(cols)
            if table:
                lines.append('| ' + ' | '.join(table[0]) + ' |')
                lines.append('|' + '|'.join([' --- ' for _ in table[0]]) + '|')
                for row in table[1:]:
                    lines.append('| ' + ' | '.join(row) + ' |')
        elif el.name == 'br':
            lines.append('')
        elif el.name == 'p':
            # Используем join по stripped_strings, чтобы не сливались слова
            lines.append(' '.join(el.stripped_strings) + '\n')
        else:
            for child in el.children:
                process(child, prefix)
    process(element)
    return '\n'.join([line for line in lines if line.strip()])


def parse_page(url: str) -> str:
    """
    Загружает страницу по url и возвращает текст главного заголовка <h1 ...> и основного содержимого <article> в markdown-формате.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        main_container = soup.find('main', class_='content-wrapper')
        if not main_container:
            print(f"Контейнер <main class='content-wrapper'> не найден на {url}")
            return ""
        section = main_container.find('section')
        if not section:
            print(f"Section не найден на {url}")
            return ""
        h1 = section.find('h1')
        article = None
        if h1:
            # Ищем article, который идёт сразу после h1
            next_sibling = h1.find_next_sibling()
            while next_sibling is not None:
                if isinstance(next_sibling, Tag) and next_sibling.name == 'article':
                    article = next_sibling
                    break
                next_sibling = next_sibling.find_next_sibling()
        if h1 and article:
            h1_md = html_to_markdown(h1)
            article_md = html_to_markdown(article)
            return f"{h1_md}\n\n{article_md}"
        else:
            print(f"h1 или article не найден на {url}")
            return ""
    except Exception as e:
        print(f"Ошибка при обработке {url}: {e}")
        return ""


def url_to_filename(url: str) -> str:
    parsed = urlparse(url)
    # Убираем протокол, заменяем / на _, убираем лишние символы
    path = parsed.path.strip('/').replace('/', '_')
    netloc = parsed.netloc.replace('.', '_')
    if path:
        filename = f"{netloc}_{path}.md"
    else:
        filename = f"{netloc}.md"
    return filename


def main():
    all_urls = spec_urls + base_urls
    output_dir = 'parsed_pages'
    os.makedirs(output_dir, exist_ok=True)
    for url in all_urls:
        print(f"Парсим: {url}")
        content = parse_page(url)
        filename = url_to_filename(url)
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)


if __name__ == "__main__":
    main()


