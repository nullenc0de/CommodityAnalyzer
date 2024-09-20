"""
Comprehensive Stock Analysis Script

This script performs an in-depth analysis of a given stock, incorporating technical,
fundamental, and news sentiment analysis. It provides an overall score, recommendation,
and detailed breakdown of various factors influencing the stock's performance.

Usage:
python script_name.py TICKER [-s]

TICKER: Stock symbol (e.g., AAPL, msft)
-s: Optional flag for summary output

Example:
python script_name.py AAPL -s

Note: This script is for educational purposes only and should not be used for actual trading
without further development and risk management considerations.

Dependencies: yfinance, pandas, numpy, matplotlib, talib, requests, beautifulsoup4
"""

import warnings
import os

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Suppress Matplotlib font manager warning
os.environ['MPLCONFIGDIR'] = os.getcwd() + "/matplotlib_config"

# Rest of the imports
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
        try:
            self.calculate_indicators()

            current_price = self.close_prices.iloc[-1]
            sma5 = self.data['SMA5'].iloc[-1]
            sma20 = self.data['SMA20'].iloc[-1]
            sma50 = self.data['SMA50'].iloc[-1]
            sma200 = self.data['SMA200'].iloc[-1]
            rsi = self.data['RSI'].iloc[-1]
            macd = self.data['MACD'].iloc[-1]
            macd_signal = self.data['MACD_Signal'].iloc[-1]
            atr = self.data['ATR'].iloc[-1]
            bb_width = self.data['BB_Width'].iloc[-1]

            avg_volume = self.volumes.mean()
            volatility = self.close_prices.pct_change().std() * np.sqrt(252)  # Annualized volatility

            recent_high = self.high_prices.iloc[-20:].max()
            recent_low = self.low_prices.iloc[-20:].min()

            conservative_exit = current_price * 1.1
            moderate_exit = current_price * 1.2
            aggressive_exit = current_price * 1.3

            potential_profit = moderate_exit - current_price
            stop_loss = current_price - atr * 2
            risk_reward_ratio = potential_profit / (current_price - stop_loss)

            self.results = {
                'trend': 'Bullish' if current_price > sma50 > sma200 else 'Bearish',
                'current_price': current_price,
                'sma5': sma5,
                'sma20': sma20,
                'sma50': sma50,
                'sma200': sma200,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'atr': atr,
                'bb_width': bb_width,
                'avg_volume': avg_volume,
                'volatility': volatility,
                'recent_high': recent_high,
                'recent_low': recent_low,
                'fifty_two_week_high': self.info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': self.info.get('fiftyTwoWeekLow'),
                'conservative_exit': conservative_exit,
                'moderate_exit': moderate_exit,
                'aggressive_exit': aggressive_exit,
                'potential_profit': potential_profit,
                'risk_reward_ratio': risk_reward_ratio,
                'stop_loss': stop_loss
            }

            self.score_components = {
                'trend': 10 if current_price > sma50 > sma200 else -10,
                'rsi': 5 if 40 < rsi < 60 else (-5 if rsi > 70 or rsi < 30 else 0),
                'macd': 5 if macd > macd_signal else -5,
                'volatility': -5 if volatility > 0.3 else (5 if volatility < 0.2 else 0),
                'risk_reward': 10 if risk_reward_ratio > 3 else (5 if risk_reward_ratio > 2 else 0)
            }

            return self.results, self.score_components
        except Exception as e:
            print(f"Error in technical analysis: {str(e)}")
            raise

class FundamentalAnalyzer(Analyzer):
    def fetch_fundamental_indicators(self, ticker: str):
        stock = yf.Ticker(ticker)
        info = stock.info

        fundamental_indicators = {
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'peg_ratio': info.get('pegRatio'),
            'price_to_book': info.get('priceToBook'),
            'debt_to_equity': info.get('debtToEquity'),
            'current_ratio': info.get('currentRatio'),
            'profit_margin': info.get('profitMargins'),
            'return_on_equity': info.get('returnOnEquity'),
            'revenue_growth': info.get('revenueGrowth'),
            'eps': info.get('trailingEps'),
            'free_cash_flow': info.get('freeCashflow')
        }

        return fundamental_indicators

    def analyze(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        try:
            self.results = self.fetch_fundamental_indicators(self.info['symbol'])

            self.score_components = {
                'pe_ratio': 10 if self.results['pe_ratio'] and self.results['pe_ratio'] < 20 else (-10 if self.results['pe_ratio'] and self.results['pe_ratio'] > 50 else 0),
                'peg_ratio': 10 if self.results['peg_ratio'] and self.results['peg_ratio'] < 1 else (-10 if self.results['peg_ratio'] and self.results['peg_ratio'] > 2 else 0),
                'price_to_book': 5 if self.results['price_to_book'] and self.results['price_to_book'] < 3 else (-5 if self.results['price_to_book'] and self.results['price_to_book'] > 5 else 0),
                'debt_to_equity': 5 if self.results['debt_to_equity'] and self.results['debt_to_equity'] < 1 else (-5 if self.results['debt_to_equity'] and self.results['debt_to_equity'] > 2 else 0),
                'current_ratio': 5 if self.results['current_ratio'] and self.results['current_ratio'] > 1.5 else (-5 if self.results['current_ratio'] and self.results['current_ratio'] < 1 else 0),
                'profit_margin': 10 if self.results['profit_margin'] and self.results['profit_margin'] > 0.2 else (-10 if self.results['profit_margin'] and self.results['profit_margin'] < 0 else 0),
                'return_on_equity': 10 if self.results['return_on_equity'] and self.results['return_on_equity'] > 0.15 else (-10 if self.results['return_on_equity'] and self.results['return_on_equity'] < 0 else 0),
                'revenue_growth': 10 if self.results['revenue_growth'] and self.results['revenue_growth'] > 0.1 else (-10 if self.results['revenue_growth'] and self.results['revenue_growth'] < 0 else 0)
            }

            return self.results, self.score_components
        except Exception as e:
            print(f"Error in fundamental analysis: {str(e)}")
            raise

class NewsSentimentAnalyzer(Analyzer):
    def __init__(self, data: pd.DataFrame, info: Dict[str, Any]):
        super().__init__(data, info)
        self.api_key = "THUTRZX4CJ4MC33O"  # Replace with your actual API key

    def analyze(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        try:
            symbol = self.info['symbol']
            news_sentiment = self.get_news_sentiment(symbol)
            
            if news_sentiment:
                sentiment_score = news_sentiment['average_score']
                sentiment_label = news_sentiment['average_label']
            else:
                sentiment_score = 0
                sentiment_label = "Neutral"

            self.results = {
                'news_sentiment': news_sentiment,
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label
            }

            self.score_components = {
                'news_sentiment': 10 * (sentiment_score - 0.5)  # Scale from -5 to 5
            }

            return self.results, self.score_components
        except Exception as e:
            print(f"Error in news sentiment analysis: {str(e)}")
            self.results = {
                'news_sentiment': {},
                'sentiment_score': 0,
                'sentiment_label': "Error"
            }
            self.score_components = {'news_sentiment': 0}
            return self.results, self.score_components

    def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={self.api_key}"
        try:
            response = requests.get(url)
            data = response.json()
            
            if 'feed' not in data:
                print(f"No news data found for {symbol}. API response: {data}")
                return {}

            feed = data['feed']
            sentiment_scores = []
            sentiment_labels = []
            
            for item in feed:
                sentiment_scores.append(float(item['overall_sentiment_score']))
                sentiment_labels.append(item['overall_sentiment_label'])
            
            if not sentiment_scores:
                print(f"No sentiment scores found for {symbol}")
                return {}

            average_score = sum(sentiment_scores) / len(sentiment_scores)
            average_label = max(set(sentiment_labels), key=sentiment_labels.count)
            
            return {
                'average_score': average_score,
                'average_label': average_label,
                'items': [{'title': item['title'], 'sentiment_score': item['overall_sentiment_score'], 'sentiment_label': item['overall_sentiment_label']} for item in feed[:5]]  # Include top 5 news items
            }
        except Exception as e:
            print(f"Error fetching news for {symbol}: {str(e)}")
            return {}

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

    def get_sector_weights(self) -> Dict[str, float]:
        sector_weights = {
            'Technology': {
                'technical': 1.2, 'fundamental': 1.1, 'news_sentiment': 1.0
            },
            'Healthcare': {
                'technical': 0.9, 'fundamental': 1.2, 'news_sentiment': 1.0
            },
            'Financials': {
                'technical': 1.0, 'fundamental': 1.3, 'news_sentiment': 1.1
            },
            'Consumer Cyclical': {
                'technical': 1.1, 'fundamental': 1.0, 'news_sentiment': 1.2
            },
            'Consumer Defensive': {
                'technical': 0.8, 'fundamental': 1.1, 'news_sentiment': 0.9
            },
            'Energy': {
                'technical': 1.2, 'fundamental': 1.0, 'news_sentiment': 1.1
            },
            'Utilities': {
                'technical': 0.7, 'fundamental': 1.2, 'news_sentiment': 0.8
            },
            'Real Estate': {
                'technical': 0.9, 'fundamental': 1.3, 'news_sentiment': 0.9
            },
            'Industrials': {
                'technical': 1.1, 'fundamental': 1.1, 'news_sentiment': 1.0
            },
            'Basic Materials': {
                'technical': 1.2, 'fundamental': 1.0, 'news_sentiment': 1.1
            },
            'Communication Services': {
                'technical': 1.1, 'fundamental': 1.1, 'news_sentiment': 1.2
            },
            'Unknown': {
                # This is the second part of the code, continuing from the get_sector_weights method
            'technical': 1.0, 'fundamental': 1.0, 'news_sentiment': 1.0
            }
        }
        return sector_weights.get(self.sector, sector_weights['Unknown'])

    def run_analysis(self) -> Dict[str, Any]:
        try:
            analyzers: List[Analyzer] = [
                TechnicalAnalyzer(self.data, self.info),
                FundamentalAnalyzer(self.data, self.info),
                NewsSentimentAnalyzer(self.data, self.info)
            ]

            for analyzer in analyzers:
                results, scores = analyzer.analyze()
                self.analysis_results.update(results)
                analyzer_type = analyzer.__class__.__name__.lower().replace('analyzer', '')
                self.score_components.update({k: v * self.weights.get(analyzer_type, 1.0) for k, v in scores.items()})

            overall_score = self.calculate_overall_score()
            recommendation, timeframe = self.get_recommendation(overall_score)

            self.analysis_results['overall'] = {
                'score': overall_score,
                'recommendation': recommendation,
                'suggested_timeframe': timeframe,
                'score_components': self.score_components,
                'sector': self.sector,
                'weights': self.weights
            }

            return self.analysis_results
        except Exception as e:
            print(f"Error in running analysis: {str(e)}")
            raise

    def calculate_overall_score(self) -> float:
        total_score = sum(self.score_components.values())
        max_possible_score = sum(abs(score) for score in self.score_components.values())
        normalized_score = (total_score + max_possible_score) / (2 * max_possible_score) * 100
        return max(0, min(100, normalized_score))

    def get_recommendation(self, score: float) -> Tuple[str, str]:
        if score >= 80:
            return "Strong Buy", "Long-term (1 year or more)"
        elif score >= 60:
            return "Buy", "Medium-term (6-12 months)"
        elif score >= 40:
            return "Hold", "Short-term (3-6 months)"
        elif score >= 20:
            return "Sell", "Consider selling"
        else:
            return "Strong Sell", "Recommend immediate sale"

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

    def get_summary(self) -> str:
        overall = self.analysis_results['overall']
        technical = self.analysis_results
        return (f"Overall Score: {overall['score']:.2f}/100, "
                f"Recommendation: {overall['recommendation']}, "
                f"Suggested Action: {overall['suggested_timeframe']}, "
                f"Conservative Exit: {technical['conservative_exit']:.2f}, "
                f"Moderate Exit: {technical['moderate_exit']:.2f}, "
                f"Aggressive Exit: {technical['aggressive_exit']:.2f}")

def analyze_stock(ticker: str, summary: bool = False) -> None:
    try:
        analyzer = AdvancedStockAnalyzer(ticker)
        result = analyzer.run_analysis()
        
        if summary:
            print(analyzer.get_summary())
        else:
            print(f"Analysis for {ticker.upper()} (${result['current_price']:.2f})")
            print(f"Sector: {result['overall']['sector']}")
            print(f"\nOverall Score: {result['overall']['score']:.2f}/100")
            print(f"Recommendation: {result['overall']['recommendation']}")
            print(f"Suggested Action: {result['overall']['suggested_timeframe']}")

            print("\nTechnical Indicators:")
            technical_indicators = ['trend', 'current_price', 'sma5', 'sma20', 'sma50', 'sma200', 'rsi', 'macd', 'macd_signal', 'atr', 'bb_width', 'avg_volume', 'volatility', 'recent_high', 'recent_low', 'fifty_two_week_high', 'fifty_two_week_low', 'conservative_exit', 'moderate_exit', 'aggressive_exit', 'potential_profit', 'risk_reward_ratio', 'stop_loss']
            for indicator in technical_indicators:
                value = result.get(indicator)
                if isinstance(value, float):
                    print(f"{indicator.replace('_', ' ').title()}: {value:.2f}")
                else:
                    print(f"{indicator.replace('_', ' ').title()}: {value}")

            print("\nFundamental Indicators:")
            fundamental_indicators = ['pe_ratio', 'forward_pe', 'peg_ratio', 'price_to_book', 'debt_to_equity', 'current_ratio', 'profit_margin', 'return_on_equity', 'revenue_growth', 'eps', 'free_cash_flow']
            for indicator in fundamental_indicators:
                value = result.get(indicator)
                if isinstance(value, float):
                    print(f"{indicator.replace('_', ' ').title()}: {value:.2f}")
                elif value is not None:
                    print(f"{indicator.replace('_', ' ').title()}: {value}")
                else:
                    print(f"{indicator.replace('_', ' ').title()}: N/A")

            print("\nNews Sentiment:")
            sentiment = result.get('news_sentiment', {})
            if isinstance(sentiment, dict) and sentiment:
                sentiment_score = sentiment.get('sentiment_score', 'N/A')
                sentiment_label = sentiment.get('sentiment_label', 'N/A')
                
                if isinstance(sentiment_score, (int, float)):
                    print(f"Average Sentiment Score: {sentiment_score:.4f}")
                else:
                    print(f"Average Sentiment Score: {sentiment_score}")
                
                print(f"Average Sentiment Label: {sentiment_label}")
                
                print("\nTop News Items:")
                for item in sentiment.get('items', []):
                    print(f"- {item['title']}")
                    item_score = item.get('sentiment_score', 'N/A')
                    if isinstance(item_score, (int, float)):
                        print(f"  Score: {item_score:.4f}, Label: {item['sentiment_label']}")
                    else:
                        print(f"  Score: {item_score}, Label: {item['sentiment_label']}")
            else:
                print("No sentiment data available. This could be due to API limitations or no recent news for this stock.")

            print("\nScore Components:")
            for component, score in result['overall']['score_components'].items():
                print(f"{component.replace('_', ' ').title()}: {score:.2f}")

            try:
                analyzer.plot_technical_indicators()
            except Exception as plot_error:
                print(f"Error plotting technical indicators: {str(plot_error)}")

            print("\nNote: This analysis is for informational purposes only and should not be considered as financial advice.")

    except Exception as e:
        print(f"Error in analyzing stock: {str(e)}")
        print("Please check if the ticker symbol is correct and try again.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stock Analysis Tool")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol")
    parser.add_argument("-s", "--summary", action="store_true", help="Display summary output")
    args = parser.parse_args()

    analyze_stock(args.ticker, args.summary)
