"""
Service for sentiment analysis of market data.
"""
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tweepy

logger = logging.getLogger(__name__)

class SentimentService:
    """Service for sentiment analysis of market data."""
    
    def __init__(self, config: Dict):
        """Initialize the service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.sentiment_cache = {}
        self.cache_expiry = timedelta(minutes=5)
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Initialize Twitter client if credentials are available
        if all(key in config['api']['twitter'] for key in ['api_key', 'api_secret', 'access_token', 'access_token_secret']):
            auth = tweepy.OAuthHandler(
                config['api']['twitter']['api_key'],
                config['api']['twitter']['api_secret']
            )
            auth.set_access_token(
                config['api']['twitter']['access_token'],
                config['api']['twitter']['access_token_secret']
            )
            self.twitter_client = tweepy.API(auth)
        else:
            self.twitter_client = None
            logger.warning("Twitter API credentials not found. Social sentiment analysis will be disabled.")
            
    async def analyze_sentiment(self, symbol: str) -> Optional[Dict]:
        """Analyze sentiment for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Optional[Dict]: Sentiment analysis results
        """
        try:
            cache_key = f"{symbol}_sentiment"
            current_time = datetime.now()
            
            # Check cache
            if cache_key in self.sentiment_cache:
                cached_data, timestamp = self.sentiment_cache[cache_key]
                if current_time - timestamp < self.cache_expiry:
                    return cached_data
            
            # Get news sentiment
            news_sentiment = await self._analyze_news_sentiment(symbol)
            
            # Get social sentiment if Twitter client is available
            social_sentiment = await self._analyze_social_sentiment(symbol) if self.twitter_client else 0.0
            
            # Calculate overall sentiment
            sentiment = self._calculate_overall_sentiment(news_sentiment, social_sentiment)
            
            # Update cache
            self.sentiment_cache[cache_key] = (sentiment, current_time)
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment for {symbol}: {str(e)}")
            return None
            
    async def _analyze_news_sentiment(self, symbol: str) -> float:
        """Analyze news sentiment for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            float: News sentiment score (-1 to 1)
        """
        try:
            # Get news articles
            news_api_key = self.config['api'].get('news_api_key')
            if not news_api_key:
                logger.warning("News API key not found. News sentiment analysis will be disabled.")
                return 0.0
                
            url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={news_api_key}"
            response = requests.get(url)
            response.raise_for_status()
            
            articles = response.json().get('articles', [])
            if not articles:
                return 0.0
                
            # Analyze sentiment for each article
            sentiments = []
            for article in articles:
                title = article.get('title', '')
                description = article.get('description', '')
                if title or description:
                    text = f"{title} {description}"
                    sentiment = self.analyzer.polarity_scores(text)
                    sentiments.append(sentiment['compound'])
                    
            return sum(sentiments) / len(sentiments) if sentiments else 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {str(e)}")
            return 0.0
            
    async def _analyze_social_sentiment(self, symbol: str) -> float:
        """Analyze social media sentiment for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            float: Social sentiment score (-1 to 1)
        """
        try:
            if not self.twitter_client:
                return 0.0
                
            # Search for tweets
            tweets = self.twitter_client.search_tweets(
                q=f"${symbol} OR #{symbol}",
                lang="en",
                count=100
            )
            
            if not tweets:
                return 0.0
                
            # Analyze sentiment for each tweet
            sentiments = []
            for tweet in tweets:
                sentiment = self.analyzer.polarity_scores(tweet.text)
                sentiments.append(sentiment['compound'])
                
            return sum(sentiments) / len(sentiments) if sentiments else 0.0
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {str(e)}")
            return 0.0
            
    def _calculate_overall_sentiment(self, news_sentiment: float, social_sentiment: float) -> Dict:
        """Calculate overall sentiment from news and social sentiment.
        
        Args:
            news_sentiment: News sentiment score
            social_sentiment: Social sentiment score
            
        Returns:
            Dict: Overall sentiment analysis
        """
        try:
            # Calculate weighted average
            news_weight = 0.7
            social_weight = 0.3
            
            # Check if we have both sentiment scores
            if news_sentiment == 0.0 and social_sentiment == 0.0:
                overall_sentiment = 0.0
            elif news_sentiment == 0.0:
                overall_sentiment = social_sentiment
            elif social_sentiment == 0.0:
                overall_sentiment = news_sentiment
            else:
                overall_sentiment = (news_sentiment * news_weight) + (social_sentiment * social_weight)
            
            # Determine sentiment trend
            if overall_sentiment > 0.2:
                trend = "BULLISH"
            elif overall_sentiment < -0.2:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"
                
            return {
                "overall": overall_sentiment,
                "news": news_sentiment,
                "social": social_sentiment,
                "trend": trend
            }
            
        except Exception as e:
            logger.error(f"Error calculating overall sentiment: {str(e)}")
            return {
                "overall": 0.0,
                "news": 0.0,
                "social": 0.0,
                "trend": "NEUTRAL"
            }
            
    def clear_cache(self):
        """Clear all cached data."""
        self.sentiment_cache.clear() 