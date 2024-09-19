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
from discord import Webhook

class StockMonitor:
    def __init__(self):
        self.symbols = []
        self.session = None
        self.error_count = {}
        self.discord_webhook = os.getenv('https://discord.com/api/webhooks/1286420702173597807/hNgcuYY68fm6t0ncWSSGt2QwrQvEybW5uRrr2nXZCMiizQnq6Wguhm41SBJcO8TicQWy')
        self.slack_webhook = os.getenv('SLACK_WEBHOOK')
        self.high_potential_stocks = []
        self.processed_count = 0
        self.total_symbols = 0
        self.monitor_interval = 3600  # Default to 1 hour
        self.is_monitoring = False

    async def get_stock_symbols(self):
        url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        reader = csv.reader(StringIO(content), delimiter='|')
                        next(reader)  # Skip the header
                        symbols = [row[0] for row in reader if row[0] != "File Creation Time" and not row[0].endswith(('W', 'R')) and "test" not in row[0].lower()]
                        self.total_symbols = len(symbols)
                        print(f"Successfully fetched {self.total_symbols} symbols")
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
        if stock_data is None or stock_data['data'].empty or len(stock_data['data']) < 20:
            return None

        df = stock_data['data']
        meta = stock_data['meta']

        current_price = df['Close'].iloc[-1]
        open_price = df['Open'].iloc[-1]

        if open_price == 0 or pd.isna(open_price) or current_price == 0 or pd.isna(current_price):
            return None

        price_change = (current_price - open_price) / open_price * 100 if open_price != current_price else 0

        total_volume = df['Volume'].sum()
        avg_volume = df['Volume'].mean()

        if total_volume < 10000 or avg_volume < 1000:
            return None

        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else None

        sma_5 = df['Close'].rolling(window=5).mean().iloc[-1]
        sma_20 = df['Close'].rolling(window=20).mean().iloc[-1]

        # Calculate ATR
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=14).mean()
        atr = df['ATR'].iloc[-1]

        # Calculate RSI
        rsi = talib.RSI(df['Close'].values, timeperiod=14)[-1]

        # Calculate MACD
        macd, signal, _ = talib.MACD(df['Close'].values)
        macd = macd[-1]
        signal = signal[-1]

        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['Close'].values, timeperiod=20)
        bb_width = (upper[-1] - lower[-1]) / middle[-1]

        recent_high = df['High'].tail(20).max()
        recent_low = df['Low'].tail(20).min()

        conservative_exit = min(current_price * 1.05, current_price + atr)
        moderate_exit = min(current_price * 1.10, current_price + 1.5 * atr)
        aggressive_exit = min(current_price * 1.15, recent_high)

        if current_price < sma_20:
            conservative_exit = min(conservative_exit, sma_20)
        if current_price < sma_5:
            moderate_exit = max(moderate_exit, sma_5)

        potential_profit = (aggressive_exit - current_price) / current_price * 100

        # Risk management calculations
        stop_loss = current_price - 2 * atr  # Set stop loss at 2 ATR below current price
        risk_per_share = current_price - stop_loss
        reward_per_share = aggressive_exit - current_price
        risk_reward_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0

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
            'macd': macd,
            'macd_signal': signal,
            'bb_width': bb_width,
            'sma_5': sma_5,
            'sma_20': sma_20,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'fifty_two_week_high': meta.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': meta.get('fiftyTwoWeekLow'),
            'conservative_exit': conservative_exit,
            'moderate_exit': moderate_exit,
            'aggressive_exit': aggressive_exit,
            'potential_profit': potential_profit,
            'stop_loss': stop_loss,
            'risk_reward_ratio': risk_reward_ratio
        }

    def rate_stock(self, metrics):
        score = 0
        reasons = []

        if metrics['price_change'] > 2:
            score += 1
            reasons.append('Price increased by more than 2%')

        if metrics['total_volume'] > 500000:
            score += 1
            reasons.append('High total volume')

        if metrics['volatility'] > 0.03:
            score += 1
            reasons.append('High volatility')

        if metrics['rsi'] < 30:
            score += 1
            reasons.append('Oversold conditions (RSI < 30)')

        if metrics['macd'] > metrics['macd_signal']:
            score += 1
            reasons.append('Bullish MACD crossover')

        if metrics['potential_profit'] > 5:
            score += 1
            reasons.append('Potential profit greater than 5%')

        return score, reasons

    async def monitor_stocks(self):
        while self.is_monitoring:
            print("Fetching stock symbols...")
            self.symbols = await self.get_stock_symbols()

            if not self.symbols:
                print("No symbols available for monitoring. Retrying in 1 hour...")
                await asyncio.sleep(self.monitor_interval)
                continue

            self.processed_count = 0
            print(f"Monitoring {self.total_symbols} stocks...")

            async with aiohttp.ClientSession() as session:
                self.session = session
                tasks = [self.monitor_stock(symbol) for symbol in self.symbols]
                await asyncio.gather(*tasks)

            print("All stocks processed. Sleeping for 1 hour before next monitoring cycle...")
            await asyncio.sleep(self.monitor_interval)

    async def monitor_stock(self, symbol):
        try:
            stock_data = await self.get_stock_data(symbol)
            if stock_data:
                metrics = self.calculate_metrics(stock_data)
                if metrics:
                    score, reasons = self.rate_stock(metrics)
                    if score >= 3:
                        print(f"High potential stock: {symbol} (Score: {score}) - {', '.join(reasons)}")
                        self.high_potential_stocks.append({
                            'symbol': symbol,
                            'score': score,
                            'reasons': reasons,
                            'metrics': metrics
                        })
                    else:
                        print(f"Stock {symbol} does not meet the criteria (Score: {score})")
        except Exception as e:
            print(f"Error processing stock {symbol}: {str(e)}")

        finally:
            self.processed_count += 1
            if self.processed_count % 100 == 0:
                print(f"Processed {self.processed_count}/{self.total_symbols} stocks")

    async def send_to_discord(self, message):
        if self.discord_webhook:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.discord_webhook, json={"content": message}) as response:
                        if response.status == 204:
                            print("Message sent to Discord")
                        else:
                            print(f"Failed to send message to Discord: HTTP {response.status}")
            except Exception as e:
                print(f"Error sending message to Discord: {str(e)}")
        else:
            print("Discord webhook URL is not set")

    async def send_to_slack(self, message):
        if self.slack_webhook:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(self.slack_webhook, json={"text": message}) as response:
                        if response.status == 200:
                            print("Message sent to Slack")
                        else:
                            print(f"Failed to send message to Slack: HTTP {response.status}")
            except Exception as e:
                print(f"Error sending message to Slack: {str(e)}")
        else:
            print("Slack webhook URL is not set")

    async def start_monitoring(self, interval=3600):
        self.monitor_interval = interval
        self.is_monitoring = True
        print("Starting stock monitoring...")
        await self.monitor_stocks()

    async def stop_monitoring(self):
        self.is_monitoring = False
        print("Stopping stock monitoring...")

if __name__ == "__main__":
    monitor = StockMonitor()
    asyncio.run(monitor.start_monitoring(3600))  # Start monitoring with a 1-hour interval
