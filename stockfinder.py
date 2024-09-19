import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import csv
from io import StringIO
import random
import traceback
from bs4 import BeautifulSoup
import requests
import talib

class StockMonitor:
    def __init__(self):
        self.symbols = []
        self.session = None
        self.error_count = {}
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK')
        self.slack_webhook = os.getenv('SLACK_WEBHOOK')
        self.high_potential_stocks = []

    async def get_stock_symbols(self):
        url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        reader = csv.reader(StringIO(content), delimiter='|')
                        next(reader)  # Skip the header
                        symbols = [row[0] for row in reader if row[0] != "File Creation Time" and not row[0].endswith(('W', 'R'))]
                        print(f"Successfully fetched {len(symbols)} symbols")
                        return symbols
                    else:
                        print(f"Failed to fetch stock symbols: HTTP {response.status}")
                        return []
        except Exception as e:
            print(f"Error in get_stock_symbols: {str(e)}")
            traceback.print_exc()
            return []

    async def get_stock_data(self, symbol):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            "range": "1y",
            "interval": "1d"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        max_retries = 5
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(0.5 + random.random())  # Add a random delay between 0.5 and 1.5 seconds
                async with self.session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self.process_stock_data(symbol, data)
                    elif response.status == 404:
                        print(f"Stock {symbol} not found on Yahoo Finance")
                        return None
                    elif response.status == 429:
                        wait_time = 2 ** attempt + random.random()
                        print(f"Rate limit hit for {symbol}. Retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        print(f"Error fetching data for {symbol}: HTTP Status {response.status}")
            except Exception as e:
                print(f"Error fetching data for {symbol}: {str(e)}")

            if attempt < max_retries - 1:
                await asyncio.sleep(1)  # Wait 1 second before retrying

        # If all attempts fail, try the failover API
        return await self.get_stock_data_failover(symbol)

    async def get_stock_data_failover(self, symbol):
        # This is a placeholder for a failover API. In a real scenario, you'd implement
        # a connection to an alternative stock data provider.
        print(f"Attempting to fetch data for {symbol} from failover API...")
        # Implement your failover logic here
        return None  # Return None if failover also fails

    def process_stock_data(self, symbol, data):
        if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
            result = data['chart']['result'][0]
            timestamps = result.get('timestamp')
            indicators = result.get('indicators', {}).get('quote', [{}])[0]

            if not timestamps or not indicators:
                print(f"Warning: Incomplete data received for {symbol}")
                return None

            df = pd.DataFrame({
                'Timestamp': pd.to_datetime(timestamps, unit='s'),
                'Open': indicators.get('open'),
                'High': indicators.get('high'),
                'Low': indicators.get('low'),
                'Close': indicators.get('close'),
                'Volume': indicators.get('volume')
            })
            df.set_index('Timestamp', inplace=True)
            df = df.dropna()  # Remove any rows with NaN values

            if not df.empty:
                return {
                    'symbol': symbol,
                    'data': df,
                    'meta': result.get('meta', {})
                }
            else:
                print(f"Warning: No valid data points for {symbol}")
        else:
            print(f"Error: Unexpected data format received for {symbol}")
        return None

    def calculate_metrics(self, stock_data):
        if stock_data is None or stock_data['data'].empty or len(stock_data['data']) < 20:  # Need at least 20 days of data
            return None

        df = stock_data['data']
        meta = stock_data['meta']

        current_price = df['Close'].iloc[-1]
        open_price = df['Open'].iloc[-1]  # Today's open price

        if open_price == 0 or pd.isna(open_price) or current_price == 0 or pd.isna(current_price):
            return None

        price_change = (current_price - open_price) / open_price * 100 if open_price != current_price else 0

        total_volume = df['Volume'].sum()  # Total volume over the period
        avg_volume = df['Volume'].mean()  # Average volume over the period

        if total_volume < 10000 or avg_volume < 1000:  # Minimum liquidity threshold
            return None

        # Calculate volatility (standard deviation of returns)
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else None

        # Simple moving averages
        sma_5 = df['Close'].rolling(window=5).mean().iloc[-1]
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]

        # Calculate Average True Range (ATR) for volatility
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        atr = df['ATR'].iloc[-1]

        # Calculate RSI
        rsi = talib.RSI(df['Close'].values, timeperiod=14)[-1]

        # Find recent support and resistance levels
        recent_high = df['High'].tail(20).max()
        recent_low = df['Low'].tail(20).min()

        # Calculate exit prices
        conservative_exit = min(current_price * 1.05, current_price + atr)  # 5% gain or 1 ATR, whichever is lower
        moderate_exit = min(current_price * 1.10, current_price + 1.5 * atr)  # 10% gain or 1.5 ATR, whichever is lower
        aggressive_exit = min(current_price * 1.15, recent_high)  # 15% gain or recent high, whichever is lower

        # Adjust exit prices based on moving averages
        if current_price < sma_20:
            conservative_exit = min(conservative_exit, sma_20)
        if current_price < sma_5:
            moderate_exit = max(moderate_exit, sma_5)

        # Calculate potential profit
        potential_profit = (aggressive_exit - current_price) / current_price * 100

        return {
            'symbol': stock_data['symbol'],
            'current_price': current_price,
            'open_price': open_price,
            'price_change': price_change,
            'total_volume': total_volume,
            'avg_volume': avg_volume,
            'volatility': volatility,
            'atr': atr,
            'rsi': rsi,
            'sma_5': sma_5,
            'sma_20': sma_20,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'fifty_two_week_high': meta.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': meta.get('fiftyTwoWeekLow'),
            'conservative_exit': conservative_exit,
            'moderate_exit': moderate_exit,
            'aggressive_exit': aggressive_exit,
            'potential_profit': potential_profit
        }

    def get_news_sentiment(self, symbol):
        url = f"https://finviz.com/quote.ashx?t={symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            news_table = soup.find(id='news-table')
            news_sentiment = {
                "positive": 0,
                "neutral": 0,
                "negative": 0
            }
            if news_table:
                for row in news_table.find_all('tr'):
                    headline = row.find_all('td')[1].get_text()
                    if "upgraded" in headline or "outperform" in headline:
                        news_sentiment["positive"] += 1
                    elif "downgraded" in headline or "underperform" in headline:
                        news_sentiment["negative"] += 1
                    else:
                        news_sentiment["neutral"] += 1
            return news_sentiment
        except Exception as e:
            print(f"Error fetching news sentiment for {symbol}: {str(e)}")
            return None

    async def process_stocks(self):
        self.session = aiohttp.ClientSession()
        self.symbols = await self.get_stock_symbols()
        for symbol in self.symbols:
            try:
                stock_data = await self.get_stock_data(symbol)
                metrics = self.calculate_metrics(stock_data)
                if metrics and metrics['potential_profit'] > 10 and metrics['volatility'] < 0.3 and metrics['rsi'] < 30:
                    sentiment = self.get_news_sentiment(symbol)
                    if sentiment and sentiment['positive'] > sentiment['negative']:
                        print(f"High potential stock found: {metrics['symbol']} with potential profit of {metrics['potential_profit']:.2f}%")
                        self.high_potential_stocks.append(metrics)
            except Exception as e:
                print(f"Error processing stock {symbol}: {str(e)}")
                traceback.print_exc()

        await self.session.close()

    async def start(self):
        await self.process_stocks()
        # Further logic to notify via Slack or Discord can be added here

if __name__ == "__main__":
    stock_monitor = StockMonitor()
    asyncio.run(stock_monitor.start())
