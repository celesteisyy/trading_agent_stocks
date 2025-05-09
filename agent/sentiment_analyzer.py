import logging
import pandas as pd
import numpy as np
import torch
from typing import Tuple, Dict, Union, List
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer

class SentimentAnalyzer:
    """
    Analyze sentiment from text data (Reddit posts) using either VADER or transformer-based models.
    Supports chunking for long texts and handles model loading/fallbacks.
    """
    
    def __init__(
        self,
        method: str = 'transformer',
        transformer_model: str = 'ProsusAI/finbert',
        chunk_stride: int = 128
    ):
        """
        Initialize sentiment analyzer.
        
        Args:
            method: 'vader' or 'transformer'
            transformer_model: Model name for transformer (e.g., 'ProsusAI/finbert')
            chunk_stride: Overlap size when windowing long texts
        """
        self.logger = logging.getLogger(__name__)
        self.method = method.lower()
        self.chunk_stride = chunk_stride
        
        if self.method == 'vader':
            try:
                # try downloading VADER dictionary
                import nltk
                try:
                    nltk.data.find('sentiment/vader_lexicon.zip')
                except LookupError:
                    self.logger.info("Downloading NLTK vader_lexicon...")
                    nltk.download('vader_lexicon', quiet=True)
                    
                self.analyzer = SentimentIntensityAnalyzer()
                self.logger.info("Initialized VADER sentiment analyzer")
            except Exception as e:
                self.logger.error(f"Failed to initialize VADER: {e}")
                self.method = 'fallback'
                
        # Initialize the appropriate analyzer
        if self.method == 'vader':
            try:
                self.analyzer = SentimentIntensityAnalyzer()
                self.logger.info("Initialized VADER sentiment analyzer")
            except Exception as e:
                self.logger.error(f"Failed to initialize VADER: {e}")
                self.method = 'fallback'
        else:
            # Initialize transformer model
            try:
                self.logger.info(f"Loading transformer model: {transformer_model}")
                self.tokenizer = AutoTokenizer.from_pretrained(transformer_model, use_fast=True)
                device = 0 if torch.cuda.is_available() else -1
                self.sentiment_pipe = pipeline(
                    "sentiment-analysis",
                    model=transformer_model,
                    tokenizer=self.tokenizer,
                    framework="pt",
                    device=device
                )
                self.logger.info(f"Successfully loaded {transformer_model}")
            except Exception as e:
                self.logger.error(f"Failed to load transformer model: {e}")
                self.method = 'fallback'
        
        # Fallback to basic sentiment
        if self.method == 'fallback':
            self.logger.warning("Using fallback sentiment analysis")
    
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment for a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            sentiment score (-1 to 1)
        """
        if not text or text.strip() == "":
            return 0.0
            
        if self.method == 'vader':
            try:
                scores = self.analyzer.polarity_scores(text)
                return scores['compound']
            except Exception as e:
                self.logger.warning(f"VADER analysis failed: {e}")
                return self._fallback_sentiment(text)
                
        elif self.method == 'transformer':
            try:
                return self._analyze_with_transformer(text)
            except Exception as e:
                self.logger.warning(f"Transformer analysis failed: {e}")
                return self._fallback_sentiment(text)
                
        else:
            return self._fallback_sentiment(text)
    
    def _analyze_with_transformer(self, text: str) -> float:
        """
        Analyze text using transformer model with chunking for long texts.
        
        Args:
            text: Text to analyze
            
        Returns:
            sentiment score (-1 to 1)
        """
        # tokenize without truncation
        toks = self.tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = toks["input_ids"][0]
        max_len = self.tokenizer.model_max_length
        stride = self.chunk_stride

        # For short texts, analyze directly
        if len(input_ids) <= max_len:
            results = self.sentiment_pipe(text, truncation=True)
            score = results[0]["score"]
            # Convert to -1 to 1 range
            if results[0]["label"].upper() in ["POSITIVE", "LABEL_1"]:
                return score
            else:
                return -score
        
        # For long texts, use windowing approach
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

        # convert each window's label+score to signed numeric
        scores = [
            (r["score"] if r["label"].upper() in ["POSITIVE", "LABEL_1"] else -r["score"])
            for r in results
        ]

        # return the average across windows
        return float(sum(scores) / len(scores))
    
    def _fallback_sentiment(self, text: str) -> float:
        """
        Simple rule-based fallback sentiment analysis.
        
        Args:
            text: Text to analyze
            
        Returns:
            sentiment score (-1 to 1)
        """
        # Simple word lists for basic sentiment
        positive_words = [
            'good', 'great', 'excellent', 'positive', 'bull', 'bullish', 'up', 'rise',
            'gain', 'profit', 'buy', 'long', 'success', 'grow', 'growth', 'increase'
        ]
        negative_words = [
            'bad', 'terrible', 'poor', 'negative', 'bear', 'bearish', 'down', 'fall',
            'loss', 'lose', 'sell', 'short', 'fail', 'drop', 'decrease', 'crash'
        ]
        
        text = text.lower()
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        total = pos_count + neg_count
        if total == 0:
            return 0.0
            
        return (pos_count - neg_count) / total
    
    def analyze_posts(self, posts_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate sentiment for each post and aggregate daily.
        
        Args:
            posts_df: DataFrame with Reddit posts (must have 'title' and 'selftext' columns)
            
        Returns:
            Tuple of (detailed_df, daily_df) with post-level and daily sentiment
        """
        self.logger.info(f"Analyzing sentiment for {len(posts_df)} posts...")
        
        # Merge title & body
        text_series = (
            posts_df.get('title', pd.Series()).fillna('') +
            " " +
            posts_df.get('selftext', pd.Series()).fillna('')
        )
        
        # Analyze each post
        sentiment_scores = []
        for text in text_series:
            score = self.analyze_text(text)
            sentiment_scores.append(score)
        
        # Combine with original data
        detailed_df = posts_df.copy()
        detailed_df['sentiment_score'] = sentiment_scores
        
        # Extract date for aggregation
        detailed_df['date'] = pd.to_datetime(detailed_df['created_utc']).dt.date
        
        # Daily average
        daily_df = (
            detailed_df
            .groupby('date')['sentiment_score']
            .agg(['mean', 'median', 'std', 'count'])
            .rename(columns={'mean': 'avg_compound'})
            .reset_index()
        )
        
        self.logger.info(f"Sentiment analysis complete. Generated {len(daily_df)} daily records.")
        return detailed_df, daily_df
    
    def correlate_with_price(self, daily_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge daily sentiment with price data and calculate correlation.
        
        Args:
            daily_df: DataFrame with daily sentiment
            price_df: DataFrame with price data
            
        Returns:
            DataFrame with merged data and correlation stats
        """
        # Ensure date formats match
        price = price_df.copy()
        sentiment = daily_df.copy()
        
        if price.index.name != 'date':
            price.index = pd.to_datetime(price.index).date
            price.index.name = 'date'
            
        sentiment['date'] = pd.to_datetime(sentiment['date'])
        
        # Join data
        merged = sentiment.set_index('date').join(
            price[['Close']].rename(columns={'Close': 'price'}),
            how='inner'
        )
        
        # Calculate correlation
        if len(merged) > 5:  # Need sufficient data points
            corr = merged['avg_compound'].corr(merged['price'])
            self.logger.info(f"Correlation between sentiment and price: {corr:.3f}")
            merged['correlation'] = corr
            
            # Calculate returns correlation
            if len(merged) > 6:  # Need at least one more point for returns
                price_ret = merged['price'].pct_change().dropna()
                sent_ret = merged['avg_compound'].diff().dropna()
                # Align the series
                aligned_df = pd.DataFrame({'price_ret': price_ret, 'sent_ret': sent_ret}).dropna()
                if len(aligned_df) > 5:
                    ret_corr = aligned_df['price_ret'].corr(aligned_df['sent_ret'])
                    self.logger.info(f"Correlation between sentiment change and price change: {ret_corr:.3f}")
                    merged['return_correlation'] = ret_corr
        
        return merged


if __name__ == '__main__':
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Test both analyzer types
    for method in ['vader', 'transformer']:
        try:
            analyzer = SentimentAnalyzer(method=method)
            
            # Test individual texts
            test_texts = [
                "I'm very bullish on this crypto project, prices will moon!",
                "This token is a complete scam, stay away and sell immediately.",
                "The market has been quite volatile lately, but holding steady."
            ]
            
            print(f"\n--- Testing {method.upper()} Analyzer ---")
            for text in test_texts:
                score = analyzer.analyze_text(text)
                print(f"Score: {score:.3f} | Text: {text[:50]}...")
            
            # Create a simple test DataFrame
            test_posts = pd.DataFrame({
                'title': test_texts,
                'selftext': ['Additional context: ' + txt for txt in test_texts],
                'created_utc': pd.date_range('2023-01-01', periods=3)
            })
            
            # Test post analysis
            detailed, daily = analyzer.analyze_posts(test_posts)
            print(f"\nDaily sentiment averages:\n{daily}")
            
        except Exception as e:
            print(f"Error testing {method}: {e}")