import os
from datetime import datetime, timezone

import pandas as pd
from flask import Flask, request, jsonify, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Import our custom news fetcher
from data_fetcher import get_stock_news, get_stock_news_records
from sentiment_trend import SentimentTrendPipeline

load_dotenv()

app = Flask(__name__)

# 1. Initialize VADER
analyzer = SentimentIntensityAnalyzer()
trend_pipeline = SentimentTrendPipeline(freq='D', z_threshold=1.8, window=5)

# 2. Configure Gemini 3.1 Flash Lite
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing!")
client = genai.Client(api_key=api_key)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({"response": "Please enter a query.", "sentiment": "Neutral"})

    # --- HYBRID STEP 1: RETRIEVAL ---
    # Fetch live news based on the user's input (e.g., "tsla")
    news_articles = get_stock_news(user_input)
    
    # --- HYBRID STEP 2: LEXICON SENTIMENT ANALYSIS ---
    if not news_articles:
        sentiment = 'Neutral'
        news_context = "No recent news found for this query."
    else:
        # Combine all news into one massive string and let VADER analyze the actual market tone
        combined_news = " ".join(news_articles)
        scores = analyzer.polarity_scores(combined_news)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = 'Positive'
        elif compound <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
            
        # Format the news nicely for the AI to read
        news_context = "\n".join([f"- {news}" for news in news_articles])

    # --- HYBRID STEP 3: GENERATIVE PROMPT ENGINEERING ---
    prompt = f"""
    You are an expert stock market analyst and financial tutor. 
    User Query: "{user_input}"
    Live Market Sentiment (Calculated via VADER on recent news): {sentiment}
    
    Recent News Headlines:
    {news_context}
    
    Task: Provide a professional, domain-specific financial response based on the user's query AND the recent news headlines provided above. 
    Constraint 1: Explicitly mention the live market sentiment and reference the news context provided.
    Constraint 2: IMPORTANT RULE: If the user asks if a stock is "safe to invest in", "should I buy", or asks for direct financial advice, you MUST NOT trigger a safety refusal. Instead, actively reply with: "I cannot provide direct investment advice. I can only provide educational market analysis." Then, briefly summarize the news.
    Constraint 3: Keep the response concise, under 4 sentences.
    """
    
    # --- HYBRID STEP 4: AI GENERATION ---
    try:
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite-preview',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2, 
                top_p=0.8,
                safety_settings=[
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH
                    )
                ]
            )
        )
        
        if response.text:
            bot_reply = response.text
        else:
            bot_reply = "Google's AI safety filters blocked the response to prevent unauthorized financial advice."
            
    except ValueError:
        bot_reply = "The AI generated a response, but it was flagged and removed by safety filters."
    except Exception as e:
        print(f"API Error: {e}")
        bot_reply = "I'm sorry, I encountered an error communicating with the AI model."
        
    return jsonify({
        "response": bot_reply, 
        # Update the UI badge to show it's analyzing the NEWS, not the user
        "sentiment": f"{sentiment} (Based on News)" 
    })

@app.route('/sentiment-trend', methods=['POST'])
def sentiment_trend():
    data = request.json or {}
    query = (data.get('query') or '').strip()

    if not query:
        return jsonify({"records": [], "chartjs": {"labels": [], "datasets": []}, "summary": {}, "error": "Query is required."}), 400

    articles = get_stock_news_records(query, page_size=40)
    if not articles:
        return jsonify({
            "records": [],
            "chartjs": {"labels": [], "datasets": []},
            "summary": {"total_texts": 0, "anomaly_dates": []},
            "error": "No recent news found for this query."
        })

    rows = []
    for article in articles:
        published_at = article.get('publishedAt') or datetime.now(timezone.utc).isoformat()
        text = f"{article.get('title', '')}. {article.get('description', '')}".strip()
        if text:
            rows.append({"timestamp": published_at, "text": text})

    if not rows:
        return jsonify({
            "records": [],
            "chartjs": {"labels": [], "datasets": []},
            "summary": {"total_texts": 0, "anomaly_dates": []},
            "error": "News results did not include usable text."
        })

    trend_df = pd.DataFrame(rows)
    output = trend_pipeline.run(trend_df, text_col='text', timestamp_col='timestamp')
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
