import warnings
import statistics
import os
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import talib
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import requests
from bs4 import BeautifulSoup
from functools import lru_cache
from typing import Dict, Tuple, List, Any
import argparse
import time
from requests.exceptions import RequestException
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

# Set matplotlib configuration directory
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), "matplotlib_config")

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)

class Analyzer(ABC):
    def __init__(self, data: pd.DataFrame, info: Dict[str, Any]):
        self.data = data
        self.info = info
        self.results: Dict[str, Any] = {}
        self.score_components: Dict[str, float] = {}

    @abstractmethod
    def analyze(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        pass

class TechnicalAnalyzer(Analyzer):
    def __init__(self, data: pd.DataFrame, info: Dict[str, Any]):
        super().__init__(data, info)
        self.close_prices = self.data['Close']
        self.high_prices = self.data['High']
        self.low_prices = self.data['Low']
        self.volumes = self.data['Volume']

    @lru_cache(maxsize=None)
    def calculate_indicators(self) -> None:
        try:
            self.data['SMA5'] = talib.SMA(self.close_prices, timeperiod=5)
            self.data['SMA20'] = talib.SMA(self.close_prices, timeperiod=20)
            self.data['SMA50'] = talib.SMA(self.close_prices, timeperiod=50)
            self.data['SMA200'] = talib.SMA(self.close_prices, timeperiod=200)
            self.data['RSI'] = talib.RSI(self.close_prices, timeperiod=14)
            self.data['MACD'], self.data['MACD_Signal'], _ = talib.MACD(self.close_prices)
            self.data['ATR'] = talib.ATR(self.high_prices, self.low_prices, self.close_prices, timeperiod=14)
            upper, middle, lower = talib.BBANDS(self.close_prices, timeperiod=20)
            self.data['BB_Width'] = (upper - lower) / middle
        except Exception as e:
            print(f"Error calculating technical indicators: {str(e)}")
            raise

    def analyze(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        self.calculate_indicators()
        
        current_price = self.close_prices.iloc[-1]
        sma50 = self.data['SMA50'].iloc[-1]
        sma200 = self.data['SMA200'].iloc[-1]
        rsi = self.data['RSI'].iloc[-1]
        macd = self.data['MACD'].iloc[-1]
        macd_signal = self.data['MACD_Signal'].iloc[-1]
        
        trend_score = 5 if current_price > sma50 > sma200 else -5
        rsi_score = 5 if 40 < rsi < 60 else (-5 if rsi > 70 or rsi < 30 else 0)
        macd_score = 5 if macd > macd_signal else -5
        
        volatility = self.close_prices.pct_change().std() * np.sqrt(252)
        volatility_score = -5 if volatility > 0.4 else (5 if volatility < 0.2 else 0)
        
        self.results = {
            'trend': 'Bullish' if current_price > sma50 > sma200 else 'Bearish',
            'current_price': current_price,
            'sma50': sma50,
            'sma200': sma200,
            'rsi': rsi,
            'macd': macd,
            'macd_signal': macd_signal,
            'volatility': volatility,
            'avg_volume': self.volumes.mean()
        }
        
        self.score_components = {
            'trend': trend_score,
            'rsi': rsi_score,
            'macd': macd_score,
            'volatility': volatility_score
        }
        
        return self.results, self.score_components

class FundamentalAnalyzer(Analyzer):
    def analyze(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        pe_ratio = self.info.get('trailingPE')
        forward_pe = self.info.get('forwardPE')
        peg_ratio = self.info.get('pegRatio')
        price_to_book = self.info.get('priceToBook')
        debt_to_equity = self.info.get('debtToEquity')
        current_ratio = self.info.get('currentRatio')
        profit_margin = self.info.get('profitMargins')
        roe = self.info.get('returnOnEquity')
        
        sector = self.info.get('sector', 'Unknown')
        pe_score = self.score_pe_ratio(pe_ratio, sector)
        peg_score = 5 if peg_ratio and peg_ratio < 1 else (-5 if peg_ratio and peg_ratio > 2 else 0)
        pb_score = 5 if price_to_book and price_to_book < 3 else (-5 if price_to_book and price_to_book > 5 else 0)
        de_score = 5 if debt_to_equity and debt_to_equity < 1 else (-5 if debt_to_equity and debt_to_equity > 2 else 0)
        cr_score = 5 if current_ratio and current_ratio > 1.5 else (-5 if current_ratio and current_ratio < 1 else 0)
        pm_score = 5 if profit_margin and profit_margin > 0.1 else (-5 if profit_margin and profit_margin < 0 else 0)
        roe_score = 5 if roe and roe > 0.15 else (-5 if roe and roe < 0 else 0)
        
        self.results = {
            'pe_ratio': pe_ratio,
            'forward_pe': forward_pe,
            'peg_ratio': peg_ratio,
            'price_to_book': price_to_book,
            'debt_to_equity': debt_to_equity,
            'current_ratio': current_ratio,
            'profit_margin': profit_margin,
            'return_on_equity': roe
        }
        
        self.score_components = {
            'pe_ratio': pe_score,
            'peg_ratio': peg_score,
            'price_to_book': pb_score,
            'debt_to_equity': de_score,
            'current_ratio': cr_score,
            'profit_margin': pm_score,
            'return_on_equity': roe_score
        }
        
        return self.results, self.score_components
    
    def score_pe_ratio(self, pe_ratio: float, sector: str) -> float:
        if not pe_ratio:
            return 0
        
        sector_pe_ranges = {
            'Technology': (15, 30),
            'Healthcare': (20, 35),
            'Financials': (10, 20),
            'Consumer Cyclical': (15, 25),
            'Consumer Defensive': (18, 28),
            'Energy': (12, 22),
            'Utilities': (16, 26),
            'Real Estate': (20, 35),
            'Industrials': (18, 28),
            'Basic Materials': (14, 24),
            'Communication Services': (18, 28)
        }
        
        low, high = sector_pe_ranges.get(sector, (15, 25))
        
        if pe_ratio < low:
            return 5
        elif pe_ratio > high:
            return -5
        else:
            return 0

class NewsSentimentAnalyzer(Analyzer):
    def __init__(self, data: pd.DataFrame, info: Dict[str, Any]):
        super().__init__(data, info)
        self.api_key = "YOUR_API_KEY_HERE"  # Replace with your actual API key

    def analyze(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        sentiment = self.get_news_sentiment(self.info['symbol'])
        
        if sentiment:
            sentiment_score = sentiment['average_score']
            sentiment_label = sentiment['average_label']
        else:
            sentiment_score = 0.5
            sentiment_label = "Neutral"

        self.results = {
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label
        }
        
        self.score_components = {
            'news_sentiment': (sentiment_score - 0.5) * 20  # Scale from -10 to 10
        }
        
        return self.results, self.score_components

    def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        try:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.api_key}"
            response = requests.get(url)
            data = response.json()
            
            if 'feed' not in data:
                return self.fallback_sentiment_analysis(symbol)

            feed = data['feed']
            sentiment_scores = []
            
            for item in feed:
                sentiment_scores.append(float(item['overall_sentiment_score']))
            
            if not sentiment_scores:
                return self.fallback_sentiment_analysis(symbol)

            average_score = sum(sentiment_scores) / len(sentiment_scores)
            average_label = "Positive" if average_score > 0.6 else "Negative" if average_score < 0.4 else "Neutral"
            
            return {
                'average_score': average_score,
                'average_label': average_label
            }
        except Exception as e:
            print(f"Error fetching news for {symbol}: {str(e)}")
            return self.fallback_sentiment_analysis(symbol)

    def fallback_sentiment_analysis(self, symbol: str) -> Dict[str, Any]:
        try:
            url = f"https://finviz.com/quote.ashx?t={symbol}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to fetch data from Finviz for {symbol}. Status code: {response.status_code}")
                return {'average_score': 0.5, 'average_label': 'Neutral'}

            soup = BeautifulSoup(response.text, 'html.parser')
            news_table = soup.find(id='news-table')
            if not news_table:
                print(f"No news table found for {symbol}")
                return {'average_score': 0.5, 'average_label': 'Neutral'}

            sia = SentimentIntensityAnalyzer()
            sentiments = []
            for row in news_table.findAll('tr'):
                title = row.a.text if row.a else ""
                if title:
                    sentiment = sia.polarity_scores(title)['compound']
                    sentiments.append(sentiment)

            if not sentiments:
                print(f"No news sentiments found for {symbol}")
                return {'average_score': 0.5, 'average_label': 'Neutral'}

            average_sentiment = sum(sentiments) / len(sentiments)
            return {
                'average_score': (average_sentiment + 1) / 2,  # Convert from [-1, 1] to [0, 1]
                'average_label': 'Positive' if average_sentiment > 0 else 'Negative' if average_sentiment < 0 else 'Neutral'
            }
        except Exception as e:
            print(f"Error in fallback sentiment analysis for {symbol}: {str(e)}")
            return {'average_score': 0.5, 'average_label': 'Neutral'}

class AdvancedStockAnalyzer:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.data = self.stock.history(period="2y")
        self.info = self.stock.info
        self.sector = self.info.get('sector', 'Unknown')
        self.analysis_results: Dict[str, Any] = {}
        self.score_components: Dict[str, float] = {}
        self.weights = self.get_sector_weights()
        self.risk_score = 0

    def get_sector_weights(self) -> Dict[str, float]:
        sector_weights = {
            'Technology': {'technical': 1.2, 'fundamental': 1.1, 'news_sentiment': 1.0},
            'Healthcare': {'technical': 0.9, 'fundamental': 1.2, 'news_sentiment': 1.0},
            'Financials': {'technical': 1.0, 'fundamental': 1.3, 'news_sentiment': 1.1},
            'Consumer Cyclical': {'technical': 1.1, 'fundamental': 1.0, 'news_sentiment': 1.2},
            'Consumer Defensive': {'technical': 0.8, 'fundamental': 1.1, 'news_sentiment': 0.9},
            'Energy': {'technical': 1.2, 'fundamental': 1.0, 'news_sentiment': 1.1},
            'Utilities': {'technical': 0.7, 'fundamental': 1.2, 'news_sentiment': 0.8},
            'Real Estate': {'technical': 0.9, 'fundamental': 1.3, 'news_sentiment': 0.9},
            'Industrials': {'technical': 1.1, 'fundamental': 1.1, 'news_sentiment': 1.0},
            'Basic Materials': {'technical': 1.2, 'fundamental': 1.0, 'news_sentiment': 1.1},
            'Communication Services': {'technical': 1.1, 'fundamental': 1.1, 'news_sentiment': 1.2},
            'Unknown': {'technical': 1.0, 'fundamental': 1.0, 'news_sentiment': 1.0}
        }
        return sector_weights.get(self.sector, sector_weights['Unknown'])

    def run_analysis(self) -> Dict[str, Any]:
        analyzers: List[Analyzer] = [
            TechnicalAnalyzer(self.data, self.info),
            FundamentalAnalyzer(self.data, self.info),
            NewsSentimentAnalyzer(self.data, self.info)
        ]

        for analyzer in analyzers:
            try:
                results, scores = analyzer.analyze()
                self.analysis_results.update(results)
                analyzer_type = analyzer.__class__.__name__.lower().replace('analyzer', '')
                self.score_components.update({k: v * self.weights.get(analyzer_type, 1.0) for k, v in scores.items()})
            except Exception as e:
                print(f"Error in {analyzer.__class__.__name__}: {str(e)}")

        overall_score = self.calculate_overall_score()
        risk_assessment = self.assess_risk()
        confidence = self.calculate_confidence_level()
        recommendation, timeframe = self.get_recommendation(overall_score, confidence, risk_assessment)

        overall_results = {
            'score': overall_score,
            'recommendation': recommendation,
            'suggested_timeframe': timeframe,
            'score_components': self.score_components,
            'sector': self.sector,
            'weights': self.weights,
            'confidence': confidence,
            'risk_assessment': risk_assessment
        }

        self.analysis_results['overall'] = overall_results
        return self.analysis_results

    def calculate_overall_score(self) -> float:
        total_weight = sum(self.weights.values())
        total_score = sum(self.score_components.values())
        
        if total_weight == 0:
            return 50  # Neutral score if no data available
        
        avg_score = total_score / total_weight
        
        # Scale to 0-100 range
        normalized_score = max(0, min(100, 50 + avg_score * 2.5))
        
        return normalized_score

    def assess_risk(self) -> str:
        volatility = self.analysis_results.get('volatility', 0)
        beta = self.info.get('beta', 1)
        debt_to_equity = self.analysis_results.get('debt_to_equity', 0)
        current_ratio = self.analysis_results.get('current_ratio', 1)

        risk_score = 0
        risk_score += 2 if volatility > 0.4 else (1 if volatility > 0.2 else 0)
        risk_score += 2 if beta > 1.5 else (1 if beta > 1 else 0)
        risk_score += 2 if debt_to_equity > 2 else (1 if debt_to_equity > 1 else 0)
        risk_score += 1 if current_ratio < 1 else 0

        self.risk_score = risk_score

        if risk_score >= 6:
            return "High Risk"
        elif risk_score >= 3:
            return "Moderate Risk"
        else:
            return "Low Risk"

    def calculate_confidence_level(self) -> float:
        data_completeness = sum(1 for v in self.analysis_results.values() if v is not None and v != 'N/A') / len(self.analysis_results)
        
        scores = [score for score in self.score_components.values() if score != 0]
        score_std = statistics.stdev(scores) if len(scores) > 1 else 0
        signal_consistency = 1 - (score_std / 10)
        
        missing_data_penalty = 0.2 * sum(1 for v in self.analysis_results.values() if v is None or v == 'N/A') / len(self.analysis_results)

        confidence = (data_completeness + signal_consistency) * 5
        confidence *= (1 - missing_data_penalty)
        confidence *= (1 - self.risk_score / 10)  # Reduce confidence for higher risk stocks

        return max(0, min(10, confidence))

    def get_recommendation(self, score: float, confidence: float, risk_assessment: str) -> Tuple[str, str]:
        if confidence < 5:
            return "Hold", "Low confidence in analysis, more research needed"
        
        risk_adjustment = 0
        if risk_assessment == "High Risk":
            risk_adjustment = 20
        elif risk_assessment == "Moderate Risk":
            risk_adjustment = 10

        adjusted_score = max(0, score - risk_adjustment)

        if adjusted_score >= 80:
            return "Strong Buy", "Long-term (1 year or more)"
        elif adjusted_score >= 60:
            return "Buy", "Medium-term (6-12 months)"
        elif adjusted_score >= 40:
            return "Hold", "Short-term (3-6 months)"
        elif adjusted_score >= 20:
            return "Sell", "Consider selling in the near term"
        else:
            return "Strong Sell", "Consider immediate sale"

    def get_summary(self) -> str:
        if 'overall' not in self.analysis_results:
            return f"Error: Unable to generate summary for {self.ticker}. Analysis incomplete."

        overall = self.analysis_results['overall']
        technical = self.analysis_results
        confidence = overall['confidence']
        score = overall['score']
        risk_assessment = overall['risk_assessment']
        
        summary = (f"{self.ticker} | Price: ${technical['current_price']:.2f} | "
                   f"Score: {score:.2f}/100 | "
                   f"Rec: {overall['recommendation']} | "
                   f"Action: {overall['suggested_timeframe']} | "
                   f"Trend: {technical['trend']} | "
                   f"Risk: {risk_assessment} | "
                   f"Confidence: {confidence:.2f}/10")
        
        if technical['trend'] == 'Bullish' and overall['recommendation'] in ['Sell', 'Strong Sell']:
            summary += " | Note: Bullish trend but other factors suggest caution"
        
        if overall['score'] < 40:
            summary += " | Warning: Low overall score, high risk"
        
        if self.analysis_results.get('volatility', 0) > 0.3:
            summary += f" | Warning: High volatility ({self.analysis_results['volatility']:.2f})"
        
        if self.analysis_results.get('avg_volume', 0) < 100000:
            summary += f" | Warning: Low trading volume ({self.analysis_results['avg_volume']:.0f})"
        
        if self.analysis_results.get('debt_to_equity', 0) > 2:
            summary += f" | Warning: High debt-to-equity ratio ({self.analysis_results['debt_to_equity']:.2f})"
        
        negative_factors = [k for k, v in self.score_components.items() if v < 0]
        if negative_factors:
            summary += f" | Warning: Negative indicators in {', '.join(negative_factors)}"

        return summary

    def plot_technical_indicators(self) -> None:
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

            # Price and SMAs
            ax1.plot(self.data.index, self.data['Close'], label='Price')
            ax1.plot(self.data.index, self.data['SMA5'], label='SMA 5')
            ax1.plot(self.data.index, self.data['SMA20'], label='SMA 20')
            ax1.plot(self.data.index, self.data['SMA50'], label='SMA 50')
            ax1.plot(self.data.index, self.data['SMA200'], label='SMA 200')
            ax1.set_title(f"{self.ticker} Price and Moving Averages")
            ax1.set_ylabel("Price")
            ax1.legend()

            # RSI
            ax2.plot(self.data.index, self.data['RSI'], label='RSI')
            ax2.axhline(y=70, color='r', linestyle='--')
            ax2.axhline(y=30, color='g', linestyle='--')
            ax2.set_title("Relative Strength Index (RSI)")
            ax2.set_ylabel("RSI")
            ax2.legend()

            # MACD
            ax3.plot(self.data.index, self.data['MACD'], label='MACD')
            ax3.plot(self.data.index, self.data['MACD_Signal'], label='Signal Line')
            ax3.set_title("Moving Average Convergence Divergence (MACD)")
            ax3.set_ylabel("MACD")
            ax3.legend()

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in plotting technical indicators: {str(e)}")

def analyze_stock(ticker: str, summary: bool = False) -> None:
    try:
        analyzer = AdvancedStockAnalyzer(ticker)
        result = analyzer.run_analysis()
        
        if summary:
            print(analyzer.get_summary())
        else:
            print(f"Analysis for {ticker.upper()} (${result['current_price']:.2f})")
            print(f"Sector: {result['overall']['sector']}")
            print(f"\nOverall Score: {result['overall']['score']:.2f}/90")
            print(f"Recommendation: {result['overall']['recommendation']}")
            print(f"Suggested Action: {result['overall']['suggested_timeframe']}")

            print("\nTechnical Indicators:")
            technical_indicators = ['trend', 'current_price', 'sma50', 'sma200', 'rsi', 'macd', 'macd_signal', 'volatility', 'avg_volume']
            for indicator in technical_indicators:
                value = result.get(indicator)
                if isinstance(value, float):
                    print(f"{indicator.replace('_', ' ').title()}: {value:.2f}")
                elif value is not None:
                    print(f"{indicator.replace('_', ' ').title()}: {value}")

            print("\nFundamental Indicators:")
            fundamental_indicators = ['pe_ratio', 'forward_pe', 'peg_ratio', 'price_to_book', 'debt_to_equity', 'current_ratio', 'profit_margin', 'return_on_equity']
            for indicator in fundamental_indicators:
                value = result.get(indicator)
                if isinstance(value, float):
                    print(f"{indicator.replace('_', ' ').title()}: {value:.2f}")
                elif value is not None:
                    print(f"{indicator.replace('_', ' ').title()}: {value}")
                else:
                    print(f"{indicator.replace('_', ' ').title()}: N/A")

            print("\nNews Sentiment:")
            sentiment = result.get('sentiment_score', 'N/A')
            sentiment_label = result.get('sentiment_label', 'N/A')
            print(f"Sentiment Score: {sentiment}")
            print(f"Sentiment Label: {sentiment_label}")

            print("\nScore Components:")
            for component, score in result['overall']['score_components'].items():
                print(f"{component.replace('_', ' ').title()}: {score:.2f}")

            print("\nAnalysis Weights:")
            for component, weight in result['overall']['weights'].items():
                print(f"{component.replace('_', ' ').title()}: {weight:.2f}")

            confidence = result['overall']['confidence']
            print(f"\nConfidence in recommendation: {confidence:.2f}/10")

            try:
                analyzer.plot_technical_indicators()
            except Exception as plot_error:
                print(f"Error plotting technical indicators: {str(plot_error)}")

            print("\nNote: This analysis is for informational purposes only and should not be considered as financial advice.")

    except Exception as e:
        print(f"Error in analyzing stock: {str(e)}")
        print("Please check if the ticker symbol is correct and try again.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Stock Analysis Tool")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol")
    parser.add_argument("-s", "--summary", action="store_true", help="Display summary output")
    args = parser.parse_args()

    analyze_stock(args.ticker, args.summary)
