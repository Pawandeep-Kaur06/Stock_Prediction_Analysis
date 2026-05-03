# MarketSentiment AI

MarketSentiment AI is a domain-specific, hybrid AI financial chatbot that provides real-time stock market sentiment analysis. It combines Natural Language Processing (NLP) with Generative AI to deliver accurate and human-like financial insights through an interactive dashboard.

 System Architecture (Hybrid AI Pipeline)

This project follows a Hybrid AI approach:

1. **Data Retrieval (RAG)**
   User queries (e.g., "TSLA") trigger real-time data fetching using NewsAPI, retrieving recent financial news articles.

2. **Sentiment Analysis (VADER)**
   The fetched data is processed using VADER (Valence Aware Dictionary and sEntiment Reasoner) to calculate sentiment polarity (Positive, Negative, Neutral).

3. **Generative AI Processing (Gemini API)**
   The sentiment score, news data, and user query are passed to Gemini 2.5 Flash to generate a meaningful financial explanation.

4. **Response Delivery**
   The final response is displayed via a chatbot interface along with sentiment insights and visualizations.

5. **Safety Guardrails**
   The system avoids providing direct financial advice and focuses on informational responses.

 Key Features

* Real-time sentiment analysis using live news data
* Sentiment classification (Positive, Negative, Neutral)
* AI-powered chatbot interaction
* Sentiment trend visualization (graph-based insights)
* Watchlist tracking for selected stocks
* Session analytics (query count, sentiment distribution)
* Secure and responsible AI response handling

 Tech Stack

* **Backend:** Python, Flask
* **Frontend:** HTML, CSS, JavaScript
* **NLP Model:** VADER
* **Generative AI:** Gemini 2.5 Flash
* **Data Source:** NewsAPI
* **Libraries:** Pandas, Requests

 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Pawandeep-Kaur06/Stock_Prediction_Analysis.git
cd Stock_Prediction_Analysis
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add API Keys

Create a `.env` file and add:

```
NEWS_API_KEY=your_newsapi_key
GEMINI_API_KEY=your_gemini_api_key
```

### 5. Run the Application

```bash
python app.py
```

### 6. Open in Browser

```
http://127.0.0.1:5000
```

 How It Works

* User enters stock query
* System fetches latest news
* Sentiment is calculated using VADER
* Gemini generates explanation
* Results displayed in chatbot + dashboard

 Limitations

* Dependent on news data quality
* Sentiment does not guarantee stock performance
* Real-time accuracy depends on API availability

 Future Scope

* Multi-stock comparison
* Real-time streaming data
* Advanced ML models (Transformers)
* Personalized financial insights

 Contributors

* Backend & AI: Sentiment Analysis, API Integration
* Frontend & UI: Dashboard, Chatbot Interface

 Conclusion

MarketSentiment AI demonstrates the practical application of Hybrid AI by combining deterministic sentiment analysis with generative AI to provide meaningful, real-time financial insights in an accessible format.
