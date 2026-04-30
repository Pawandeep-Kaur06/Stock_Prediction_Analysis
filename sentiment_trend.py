"""
Sentiment Trend Over Time Pipeline
====================================
Production-ready system for aggregating stock-related text sentiment
into time-series data suitable for financial charting.

Author: Senior NLP + Data Engineering pattern
Dependencies: pandas, numpy, vaderSentiment
"""

import json
import warnings
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 1. SENTIMENT ENGINE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class SentimentEngine:
    """
    Wraps VADER with a clean score â†’ label â†’ numeric conversion.
    VADER is preferred for financial text: it handles negations,
    punctuation emphasis, and financial slang well without retraining.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def score(self, text: str) -> dict:
        """
        Returns compound score, label, and numeric value.

        Compound thresholds (VADER recommended):
          >= +0.05  â†’ Positive  â†’ +1
          <= -0.05  â†’ Negative  â†’ -1
          else      â†’ Neutral   â†’  0
        """
        if not text or not isinstance(text, str):
            return {"compound": 0.0, "label": "neutral", "numeric": 0}

        scores = self.analyzer.polarity_scores(text.strip())
        compound = round(scores["compound"], 4)

        if compound >= 0.05:
            label, numeric = "positive", 1
        elif compound <= -0.05:
            label, numeric = "negative", -1
        else:
            label, numeric = "neutral", 0

        return {
            "compound": compound,   # Raw VADER score: [-1.0, +1.0]
            "label": label,         # Human-readable class
            "numeric": numeric,     # Discrete: -1, 0, +1
        }

    def score_dataframe(self, df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
        """Vectorised scoring â€” applies score() across all rows."""
        scored = df[text_col].apply(self.score)
        df = df.copy()
        df["compound"]  = scored.apply(lambda x: x["compound"])
        df["label"]     = scored.apply(lambda x: x["label"])
        df["numeric"]   = scored.apply(lambda x: x["numeric"])
        return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 2. TIME-SERIES AGGREGATOR
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class SentimentAggregator:
    """
    Groups scored records by a time bucket (daily / hourly) and
    computes statistics useful for financial chart overlays.
    """

    def aggregate(
        self,
        df: pd.DataFrame,
        freq: Literal["D", "h"] = "D",
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        df           : DataFrame with [timestamp, compound, label, numeric]
        freq         : 'D' = daily, 'h' = hourly
        timestamp_col: column holding datetime values

        Returns
        -------
        Aggregated DataFrame with one row per time bucket, sorted ascending.
        """
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.set_index(timestamp_col).sort_index()

        agg = df.resample(freq).agg(
            avg_sentiment  = ("compound",  "mean"),   # Mean VADER compound
            numeric_mean   = ("numeric",   "mean"),   # Mean of -1/0/+1
            total_count    = ("compound",  "count"),
            positive_count = ("label", lambda s: (s == "positive").sum()),
            negative_count = ("label", lambda s: (s == "negative").sum()),
            neutral_count  = ("label", lambda s: (s == "neutral").sum()),
            sentiment_std  = ("compound",  "std"),    # Volatility proxy
        ).dropna(subset=["avg_sentiment"])

        agg["avg_sentiment"] = agg["avg_sentiment"].round(4)
        agg["numeric_mean"]  = agg["numeric_mean"].round(4)
        agg["sentiment_std"] = agg["sentiment_std"].fillna(0).round(4)

        # Dominant label per bucket
        agg["dominant"] = agg[["positive_count", "negative_count", "neutral_count"]].idxmax(axis=1)
        agg["dominant"] = agg["dominant"].str.replace("_count", "")

        return agg.reset_index()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 3. ANOMALY / SPIKE DETECTOR
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class AnomalyDetector:
    """
    Detects sentiment spikes â€” sudden shifts that often precede
    price moves. Uses Z-score over a rolling window.

    Financial relevance:
      A Z-score spike > Â±2 on sentiment often correlates with:
      - Earnings surprises
      - Regulatory announcements
      - Executive changes
      - Macro events (Fed rate decisions, CPI prints)
    """

    def __init__(self, z_threshold: float = 2.0, window: int = 7):
        """
        z_threshold : standard deviations beyond which a point is anomalous
        window      : rolling window for mean/std computation
        """
        self.z_threshold = z_threshold
        self.window = window

    def detect(self, agg_df: pd.DataFrame, score_col: str = "avg_sentiment") -> pd.DataFrame:
        """
        Adds z_score and is_anomaly columns to aggregated DataFrame.
        """
        df = agg_df.copy()
        rolling_mean = df[score_col].rolling(self.window, min_periods=2).mean().shift(1)
        rolling_std = df[score_col].rolling(self.window, min_periods=2).std().shift(1)
        rolling_std = rolling_std.replace(0, np.nan)

        z_score = (df[score_col] - rolling_mean) / rolling_std
        df["z_score"] = z_score.replace([np.inf, -np.inf], np.nan).fillna(0).round(3)
        df["is_anomaly"] = df["z_score"].abs() >= self.z_threshold
        df["anomaly_dir"] = np.where(
            df["is_anomaly"] & (df["z_score"] > 0), "spike_positive",
            np.where(df["is_anomaly"] & (df["z_score"] < 0), "spike_negative", "normal")
        )
        return df


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 4. JSON SERIALISER (Chart-ready)
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ChartSerializer:
    """
    Converts the aggregated + anomaly-annotated DataFrame into
    clean JSON structures ready for Chart.js / Recharts / D3.
    """

    def to_line_chart(
        self,
        df: pd.DataFrame,
        freq: Literal["D", "h"] = "D",
    ) -> list[dict]:
        """
        Primary output format.

        Example record:
        {
          "date": "2026-04-20",
          "sentiment_score": 0.6,
          "numeric_mean": 0.5,
          "total_count": 12,
          "positive_count": 8,
          "negative_count": 2,
          "neutral_count": 2,
          "dominant": "positive",
          "sentiment_std": 0.18,
          "z_score": 1.4,
          "is_anomaly": false,
          "anomaly_dir": "normal"
        }
        """
        ts_col = df.columns[0]  # first col is the resampled timestamp
        date_fmt = "%Y-%m-%d" if freq == "D" else "%Y-%m-%dT%H:%M"

        records = []
        for _, row in df.iterrows():
            records.append({
                "date":            row[ts_col].strftime(date_fmt),
                "sentiment_score": round(float(row["avg_sentiment"]), 4),
                "numeric_mean":    round(float(row["numeric_mean"]), 4),
                "total_count":     int(row["total_count"]),
                "positive_count":  int(row["positive_count"]),
                "negative_count":  int(row["negative_count"]),
                "neutral_count":   int(row["neutral_count"]),
                "dominant":        str(row["dominant"]),
                "sentiment_std":   round(float(row["sentiment_std"]), 4),
                "z_score":         round(float(row["z_score"]), 3),
                "is_anomaly":      bool(row["is_anomaly"]),
                "anomaly_dir":     str(row["anomaly_dir"]),
            })
        return records

    def to_recharts_multi_series(self, records: list[dict]) -> list[dict]:
        """
        Recharts-ready format with all series in one object per date.
        Drop-in for <ComposedChart> or <LineChart> with multiple <Line>.
        """
        return records  # already in the right shape for Recharts

    def to_chartjs(self, records: list[dict]) -> dict:
        """
        Chart.js dataset format.
        Plug directly into `new Chart(ctx, { data: ... })`.
        """
        labels = [r["date"] for r in records]
        return {
            "labels": labels,
            "datasets": [
                {
                    "label": "Sentiment Score",
                    "data": [r["sentiment_score"] for r in records],
                    "borderColor": "#00d4a8",
                    "backgroundColor": "rgba(0,212,168,0.08)",
                    "tension": 0.4,
                    "fill": True,
                    "pointRadius": [6 if r["is_anomaly"] else 3 for r in records],
                    "pointBackgroundColor": [
                        "#f0636b" if r["anomaly_dir"] == "spike_negative"
                        else "#00d4a8" if r["anomaly_dir"] == "spike_positive"
                        else "#5b9df0"
                        for r in records
                    ],
                },
                {
                    "label": "Numeric Mean",
                    "data": [r["numeric_mean"] for r in records],
                    "borderColor": "#5b9df0",
                    "borderDash": [5, 5],
                    "tension": 0.4,
                    "fill": False,
                    "pointRadius": 0,
                },
            ],
        }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 5. END-TO-END PIPELINE
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class SentimentTrendPipeline:
    """
    Orchestrates the full ETL:
      raw text + timestamps
        â†’ sentiment scoring
        â†’ time aggregation
        â†’ anomaly detection
        â†’ chart-ready JSON
    """

    def __init__(
        self,
        freq: Literal["D", "h"] = "D",
        z_threshold: float = 2.0,
        window: int = 7,
    ):
        self.freq       = freq
        self.engine     = SentimentEngine()
        self.aggregator = SentimentAggregator()
        self.detector   = AnomalyDetector(z_threshold=z_threshold, window=window)
        self.serializer = ChartSerializer()

    def run(
        self,
        df: pd.DataFrame,
        text_col: str = "text",
        timestamp_col: str = "timestamp",
    ) -> dict:
        """
        Parameters
        ----------
        df            : Raw DataFrame with at least [text_col, timestamp_col]
        text_col      : Column containing news headlines / tweet text
        timestamp_col : Column containing datetime strings or objects

        Returns
        -------
        {
          "records": [...],          # Primary JSON for plotting
          "chartjs": {...},          # Chart.js dataset object
          "summary": {...},          # High-level stats
        }
        """
        # Step 1 â€” Score
        scored = self.engine.score_dataframe(df, text_col=text_col)

        # Step 2 â€” Aggregate
        agg = self.aggregator.aggregate(scored, freq=self.freq, timestamp_col=timestamp_col)

        # Step 3 â€” Anomalies
        agg = self.detector.detect(agg, score_col="avg_sentiment")

        # Step 4 â€” Serialise
        records = self.serializer.to_line_chart(agg, freq=self.freq)
        chartjs  = self.serializer.to_chartjs(records)

        # Step 5 â€” Summary
        summary = self._summary(scored, records)

        return {"records": records, "chartjs": chartjs, "summary": summary}

    def _summary(self, scored: pd.DataFrame, records: list[dict]) -> dict:
        total = len(scored)
        if total == 0 or not records:
            return {
                "total_texts": 0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "overall_avg": 0.0,
                "overall_std": 0.0,
                "anomaly_dates": [],
                "most_positive_day": None,
                "most_negative_day": None,
            }

        overall_std = scored["compound"].std()
        if pd.isna(overall_std):
            overall_std = 0.0

        return {
            "total_texts":       total,
            "positive_pct":      round((scored["label"] == "positive").sum() / total * 100, 1),
            "negative_pct":      round((scored["label"] == "negative").sum() / total * 100, 1),
            "neutral_pct":       round((scored["label"] == "neutral").sum()  / total * 100, 1),
            "overall_avg":       round(scored["compound"].mean(), 4),
            "overall_std":       round(float(overall_std), 4),
            "anomaly_dates":     [r["date"] for r in records if r["is_anomaly"]],
            "most_positive_day": max(records, key=lambda r: r["sentiment_score"])["date"],
            "most_negative_day": min(records, key=lambda r: r["sentiment_score"])["date"],
        }


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# 6. DEMO RUNNER
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def generate_sample_data(n: int = 120) -> pd.DataFrame:
    """
    Simulates 120 stock-related headlines over 30 days with realistic
    sentiment distribution (biased slightly positive like real markets).
    """
    headlines = [
        # Positive
        "NVDA beats earnings expectations by wide margin, stock surges",
        "Fed signals pause in rate hikes, markets rally strongly",
        "Apple reports record iPhone sales, revenue guidance raised",
        "Strong jobs report boosts investor confidence in economy",
        "Tesla delivery numbers exceed analyst forecasts this quarter",
        "Inflation data cools, sparking broad market optimism",
        "Microsoft Azure growth accelerates, cloud dominance expands",
        "Berkshire Hathaway increases stake, signaling bullish outlook",
        "S&P 500 hits all-time high amid strong earnings season",
        "Semiconductor sector surges on AI demand tailwinds",
        # Negative
        "Bank sector tumbles amid renewed recession fears",
        "TSLA misses delivery targets, shares fall sharply",
        "Layoffs spike at major tech firms, economic uncertainty grows",
        "CPI data hotter than expected, rate cut hopes dashed",
        "China tensions escalate, global supply chain fears resurface",
        "Meta faces antitrust probe, regulatory risk spikes",
        "Crude oil crash hammers energy stocks across the board",
        "Credit card delinquency rates hit 10-year high",
        "Yield curve inversion deepens, recession signal flashes red",
        "Major hedge fund liquidates positions amid margin calls",
        # Neutral
        "Fed holds rates steady as expected, signals data dependency",
        "Earnings season begins with mixed results across sectors",
        "Markets flat as investors await key jobs report Friday",
        "Bitcoin consolidates near support, traders watch closely",
        "Quarterly GDP growth in line with consensus estimates",
        "Analyst upgrades AAPL to buy with unchanged price target",
        "Volume light ahead of holiday weekend, indices drift",
        "FOMC minutes reveal divided committee on future rate path",
    ]

    rng = np.random.default_rng(42)
    base = datetime(2026, 4, 1)
    timestamps = [base + timedelta(hours=int(h)) for h in rng.uniform(0, 30 * 24, n)]
    texts = rng.choice(headlines, size=n)

    return pd.DataFrame({"timestamp": timestamps, "text": texts})


if __name__ == "__main__":
    print("=" * 60)
    print("  Sentiment Trend Pipeline â€” Demo Run")
    print("=" * 60)

    # 1. Load data (replace with your real DataFrame here)
    df = generate_sample_data(n=200)
    print(f"\n[1] Dataset: {len(df)} records, {df['timestamp'].min().date()} â†’ {df['timestamp'].max().date()}")

    # 2. Run pipeline
    pipeline = SentimentTrendPipeline(freq="D", z_threshold=1.8, window=5)
    output = pipeline.run(df, text_col="text", timestamp_col="timestamp")

    # 3. Print summary
    s = output["summary"]
    print(f"\n[2] Summary")
    print(f"    Positive : {s['positive_pct']}%")
    print(f"    Negative : {s['negative_pct']}%")
    print(f"    Neutral  : {s['neutral_pct']}%")
    print(f"    Overall avg VADER : {s['overall_avg']}")
    print(f"    Anomaly dates     : {s['anomaly_dates'] or 'None detected'}")
    print(f"    Most positive day : {s['most_positive_day']}")
    print(f"    Most negative day : {s['most_negative_day']}")

    # 4. Print first 5 JSON records
    print(f"\n[3] First 5 chart records (JSON):")
    print(json.dumps(output["records"][:5], indent=2))

    # 5. Save full output
    with open("sentiment_trend_output.json", "w") as f:
        json.dump(output["records"], f, indent=2)
    print("\n[4] Full JSON saved â†’ sentiment_trend_output.json")
    print("\n[5] Chart.js dataset keys:", list(output["chartjs"].keys()))

    # 6. Optional: integrate into Flask route
    print("""
[6] Flask integration snippet:
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from sentiment_trend import SentimentTrendPipeline
import pandas as pd

pipeline = SentimentTrendPipeline(freq='D')

@app.route('/sentiment-trend')
def sentiment_trend():
    df = load_your_data()   # your existing news fetcher
    output = pipeline.run(df, text_col='text', timestamp_col='timestamp')
    return jsonify(output['records'])
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
""")

