import os

import requests
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API_KEY')


def get_stock_news_records(query, page_size=5):
    if not NEWS_API_KEY:
        print("Error: NEWS_API_KEY missing from .env file.")
        return []

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=12)
        response.raise_for_status()
        data = response.json()
        return data.get('articles', [])
    except Exception as e:
        print(f"News fetch error: {e}")
        return []


def get_stock_news(query):
    articles = get_stock_news_records(query, page_size=5)
    news_texts = []

    for article in articles:
        title = article.get('title') or ''
        description = article.get('description') or ''
        text = f"{title}. {description}".strip()
        if text:
            news_texts.append(text)

    return news_texts
