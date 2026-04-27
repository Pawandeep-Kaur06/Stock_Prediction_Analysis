import os
import requests
from dotenv import load_dotenv

load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

def get_stock_news(query):
    if not NEWS_API_KEY:
        print("Error: NEWS_API_KEY missing from .env file.")
        return []
    
    # Fetching the top 5 most recent English articles about the query
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    
    # Disguise the script as a web browser to prevent NewsAPI from blocking us
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        articles = data.get('articles', [])
        
        news_texts = []
        for article in articles[:5]: # Grab the top 5 articles
            title = article.get('title') or ''
            description = article.get('description') or ''
            news_texts.append(f"{title}. {description}")
            
        return news_texts
    except Exception as e:
        print(f"❌ News fetch error: {e}")
        return []