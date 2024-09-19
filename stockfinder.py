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
        self.discord_webhook = os.getenv('https://discord.com/api/webhooks/blah')
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

    def format_stock_info(self, metrics):
        return f"""
High Potential Stock Found:

Symbol: {metrics['symbol']}
Current Price: ${metrics['current_price']:.2f}
Potential Profit: {metrics['potential_profit']:.2f}%
Entry Price: ${metrics['current_price']:.2f}
Exit Price: ${metrics['aggressive_exit']:.2f}
Stop Loss: ${metrics['stop_loss']:.2f}
Risk-Reward Ratio: {metrics['risk_reward_ratio']:.2f}
Support Level: ${metrics['recent_low']:.2f}
Resistance Level: ${metrics['recent_high']:.2f}
RSI: {metrics['rsi']:.0f}
MACD: {metrics['macd']:.2f}
MACD Signal: {metrics['macd_signal']:.2f}
Bollinger Bandwidth: {metrics['bb_width']:.2f}
50-Day MA: ${metrics['sma_20']:.2f}
200-Day MA: ${metrics['sma_5']:.2f}
"""

    def format_buy_alert(self, metrics):
        return f"""
Buy Alert: {metrics['symbol']}
Entry Price: ${metrics['current_price']:.2f}
Exit Price: ${metrics['aggressive_exit']:.2f}
Stop Loss: ${metrics['stop_loss']:.2f}
"""

    def format_sell_alert(self, metrics):
        profit = metrics['aggressive_exit'] - metrics['current_price']
        profit_percentage = (profit / metrics['current_price']) * 100
        return f"""
Sell Alert: {metrics['symbol']}
Profit: ${profit:.2f} ({profit_percentage:.2f}%)
"""

    def explain_alert(self, metrics, sentiment):
        reasons = []
        if metrics['potential_profit'] > 10:
            reasons.append(f"The potential profit of {metrics['potential_profit']:.2f}% is significant.")
        if metrics['volatility'] < 0.3:
            reasons.append(f"The stock's volatility ({metrics['volatility']:.2f}) is relatively low, suggesting stability.")
        if metrics['rsi'] < 30:
            reasons.append(f"The RSI of {metrics['rsi']:.0f} indicates the stock may be oversold.")
        if metrics['macd'] > metrics['macd_signal']:
            reasons.append("The MACD is above its signal line, suggesting bullish momentum.")
        if metrics['current_price'] > metrics['sma_20'] > metrics['sma_5']:
            reasons.append("The price is above both the 50-day and 200-day moving averages, indicating an uptrend.")
        if metrics['risk_reward_ratio'] > 2:
            reasons.append(f"The risk-reward ratio of {metrics['risk_reward_ratio']:.2f} is favorable.")
        if sentiment['positive'] > sentiment['negative']:
            reasons.append("Recent news sentiment is predominantly positive.")

        explanation = "This stock is being flagged as a high potential opportunity because:\n"
        explanation += "\n".join(f"- {reason}" for reason in reasons)
        return explanation

    async def send_discord_message(self, message):
        if self.discord_webhook:
            async with aiohttp.ClientSession() as session:
                webhook = Webhook.from_url(self.discord_webhook, session=session)
                await webhook.send(content=message)

    async def send_slack_message(self, message):
        if self.slack_webhook:
            async with aiohttp.ClientSession() as session:
                await session.post(self.slack_webhook, json={"text": message})

    async def continuous_monitor(self):
        self.is_monitoring = True
        while self.is_monitoring:
            print("Starting a new monitoring cycle...")
            await self.process_stocks()
            print(f"Monitoring cycle complete. Waiting for {self.monitor_interval} seconds before next cycle.")
            await asyncio.sleep(self.monitor_interval)

    async def process_stocks(self):
        self.session = aiohttp.ClientSession()
        self.symbols = await self.get_stock_symbols()
        self.processed_count = 0
        self.high_potential_stocks = []

        for symbol in self.symbols:
            try:
                self.processed_count += 1
                if self.processed_count % 100 == 0:
                    print(f"Processed {self.processed_count}/{self.total_symbols} symbols")

                stock_data = await self.get_stock_data(symbol)
                metrics = self.calculate_metrics(stock_data)

                if metrics:
                    print(f"Debug: {symbol} - Potential Profit: {metrics['potential_profit']:.2f}%, Volatility: {metrics['volatility']:.2f}, RSI: {metrics['rsi']:.0f}, Risk-Reward Ratio: {metrics['risk_reward_ratio']:.2f}")

                    if metrics['potential_profit'] > 10 and metrics['volatility'] < 0.3 and metrics['rsi'] < 30 and metrics['risk_reward_ratio'] > 2:
                        sentiment = self.get_news_sentiment(symbol)
                        if sentiment and sentiment['positive'] > sentiment['negative']:
                            stock_info = self.format_stock_info(metrics)
                            buy_alert = self.format_buy_alert(metrics)
                            sell_alert = self.format_sell_alert(metrics)
                            explanation = self.explain_alert(metrics, sentiment)

                            full_message = f"{stock_info}\n{buy_alert}\n{sell_alert}\n\nExplanation:\n{explanation}"

                            # Print to terminal
                            print(full_message)

                            # Send to Discord and Slack
                            await self.send_discord_message(full_message)
                            await self.send_slack_message(full_message)

                            self.high_potential_stocks.append(metrics)
                    else:
                        print(f"Debug: {symbol} did not meet all criteria for high potential")
                else:
                    print(f"Debug: Unable to calculate metrics for {symbol}")
            except Exception as e:
                print(f"Error processing stock {symbol}: {str(e)}")
                traceback.print_exc()

        await self.session.close()
        print(f"Processing complete. Analyzed {self.processed_count} stocks.")
        print(f"Found {len(self.high_potential_stocks)} high potential stocks.")

        if not self.high_potential_stocks and not self.is_monitoring:
            print("No high potential stocks found. Starting continuous monitoring...")
            await self.continuous_monitor()

    async def start(self, mode='once'):
        start_time = time.time()
        if mode == 'monitor':
            await self.continuous_monitor()
        else:
            await self.process_stocks()
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Total execution time: {execution_time:.2f} seconds")

    def stop_monitoring(self):
        self.is_monitoring = False
        print("Stopping monitoring mode after the current cycle completes.")

# Main execution block
if __name__ == "__main__":
    stock_monitor = StockMonitor()

    # You can change this to 'monitor' to start in continuous monitoring mode
    mode = 'once'

    asyncio.run(stock_monitor.start(mode))
