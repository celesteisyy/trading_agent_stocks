import os

# ─── Silence oneDNN & TF INFO logs ─────────────────────────────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"]     = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "2"

import pandas as pd
import matplotlib.pyplot as plt
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer
import tensorflow as tf
from data_loader import DataLoader


class SentimentAnalyzer:
    """
    Compute sentiment scores for Reddit posts using either VADER or
    Transformer-based models (with chunking to handle >512 tokens).

    Args:
        method (str): 'vader' or 'transformer'.
        transformer_model (str): HF model name (e.g. 'ProsusAI/finbert').
        chunk_stride (int): overlap size when windowing long texts.
    """
    def __init__(
        self,
        method: str = 'vader',
        transformer_model: str = 'ProsusAI/finbert',
        chunk_stride: int = 128
    ):
        self.method = method.lower()
        if self.method == 'vader':
            self.analyzer = SentimentIntensityAnalyzer()
        else:
            # load a fast tokenizer so we can slice into token windows
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_model, use_fast=True)
            device = 0 if torch.cuda.is_available() else -1
            self.sentiment_pipe = pipeline(
                "sentiment-analysis",
                model=transformer_model,
                tokenizer=self.tokenizer,
                framework="pt",
                device=device
            )
            # how much overlap between windows
            self.chunk_stride = chunk_stride

    def _chunk_and_score(self, text: str) -> float:
        # tokenize *without* truncation
        toks = self.tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = toks["input_ids"][0]
        max_len = self.tokenizer.model_max_length
        stride = self.chunk_stride

        # build windows of size max_len with given stride
        windows = [
            input_ids[i : i + max_len]
            for i in range(0, len(input_ids), max_len - stride)
        ]

        # decode each window back to text
        texts = [
            self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for w in windows
        ]

        # infer on all windows
        results = self.sentiment_pipe(
            texts,
            truncation=True,
            max_length=max_len,
            batch_size=8
        )

        # convert each window’s label+score to signed numeric
        scores = [
            (r["score"] if r["label"].upper() in ["POSITIVE", "LABEL_1"] else -r["score"])
            for r in results
        ]

        # return the average across windows
        return float(sum(scores) / len(scores))

    def analyze_posts(self, posts_df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        """
        Calculate sentiment for each post and aggregate daily.

        Returns:
            detailed_df: posts_df + numeric_score (and VADER cols if chosen)
            daily_df: date | avg_compound
        """
        # merge title & body
        text_series = (
            posts_df.get('title', pd.Series()).fillna('') +
            " " +
            posts_df.get('selftext', pd.Series()).fillna('')
        )

        if self.method == 'vader':
            # classic VADER
            scores = text_series.apply(self.analyzer.polarity_scores)
            scores_df = pd.DataFrame(list(scores))
            scores_df['numeric_score'] = scores_df['compound']
        else:
            # transformer + chunking
            numeric_scores = text_series.apply(self._chunk_and_score)
            scores_df = pd.DataFrame({'numeric_score': numeric_scores})

        # combine, extract date
        detailed_df = pd.concat(
            [posts_df.reset_index(drop=True), scores_df],
            axis=1
        )
        detailed_df['date'] = pd.to_datetime(detailed_df['created_utc']).dt.date

        # daily average
        daily_df = (
            detailed_df
            .groupby('date')['numeric_score']
            .mean()
            .rename('avg_compound')
            .reset_index()
        )

        return detailed_df, daily_df

    def plot_daily_sentiment(self, daily_df: pd.DataFrame):
        """Plot daily avg sentiment."""
        plt.figure(figsize=(10, 4))
        plt.plot(daily_df['date'], daily_df['avg_compound'], marker='o')
        plt.title('Daily Average Reddit Sentiment')
        plt.xlabel('Date')
        plt.ylabel('Average Sentiment Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def correlate_with_price(self, daily_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge daily sentiment with price data and print correlation.
        """
        price = price_df.copy()
        if price.index.name != 'date':
            price.index = pd.to_datetime(price.index).date
            price.index.name = 'date'
        merged = daily_df.set_index('date').join(
            price[['close']].rename(columns={'close': 'price'})
        ).dropna()
        corr = merged['avg_compound'].corr(merged['price'])
        print(f"Correlation between sentiment and price: {corr:.2f}")
        return merged


if __name__ == '__main__':
    # Example end‑to‑end test
    loader = DataLoader(start_date="2021-01-01", end_date="2025-05-01")
    reddit_df = loader.load_reddit_range(
        subreddit="CryptoCurrency",
        after="2021-01-01",
        before="2025-05-01",
        max_posts=500
    )

    sa = SentimentAnalyzer(method='transformer')
    detailed_df, daily_df = sa.analyze_posts(reddit_df)

    # Plot if you want the chart
    # sa.plot_daily_sentiment(daily_df)

    # ── Print the daily sentiment as a Markdown table ────────────────────────
    print("\nDaily Average Reddit Sentiment\n")
    # to_markdown requires pandas>=1.0; it prints a GitHub‑style table
    print(daily_df.to_markdown(index=False))