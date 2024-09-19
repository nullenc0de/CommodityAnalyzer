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

class StockMonitor:
    def __init__(self):
        self.symbols = []
        self.session = None
        self.error_count = {}
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK')
        self.slack_webhook = os.getenv('SLACK_WEBHOOK')

    async def get_stock_symbols(self):
        url = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    reader = csv.reader(StringIO(content), delimiter='|')
                    next(reader)  # Skip the header
                    symbols = [row[0] for row in reader if row[0] != "File Creation Time" and not row[0].endswith(('W', 'R'))]
                    return symbols
                else:
                    print(f"Failed to fetch stock symbols: HTTP {response.status}")
                    return []

    async def get_stock_data(self, symbol):
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        params = {
            "range": "1mo",
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

        self.error_count[symbol] = self.error_count.get(symbol, 0) + 1
        return None

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

        return {
            'symbol': stock_data['symbol'],
            'current_price': current_price,
            'open_price': open_price,
            'price_change': price_change,
            'total_volume': total_volume,
            'avg_volume': avg_volume,
            'volatility': volatility,
            'atr': atr,
            'sma_5': sma_5,
            'sma_20': sma_20,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'fifty_two_week_high': meta.get('fiftyTwoWeekHigh'),
            'fifty_two_week_low': meta.get('fiftyTwoWeekLow'),
            'conservative_exit': conservative_exit,
            'moderate_exit': moderate_exit,
            'aggressive_exit': aggressive_exit
        }

    async def send_discord_alert(self, message):
        if not self.discord_webhook:
            return

        async with aiohttp.ClientSession() as session:
            webhook = {'content': message}
            async with session.post(self.discord_webhook, json=webhook) as response:
                if response.status != 204:
                    print(f"Failed to send Discord alert: HTTP {response.status}")

    async def send_slack_alert(self, message):
        if not self.slack_webhook:
            return

        async with aiohttp.ClientSession() as session:
            webhook = {'text': message}
            async with session.post(self.slack_webhook, json=webhook) as response:
                if response.status != 200:
                    print(f"Failed to send Slack alert: HTTP {response.status}")

    async def monitor_stock(self, symbol):
        stock_data = await self.get_stock_data(symbol)
        if stock_data:
            metrics = self.calculate_metrics(stock_data)
            if metrics and metrics['current_price'] > 0 and metrics['open_price'] > 0:
                price_change_threshold = 5  # Alert if price increase is more than 5%
                volume_increase_threshold = 2  # Alert if volume is 2 times the average volume
                min_price = 5  # Minimum price threshold ($5)
                max_price = 1000  # Maximum price threshold ($1000)
                min_volume = 100000  # Minimum volume threshold

                is_interesting = False
                reasons = []

                current_price = metrics['current_price']
                if min_price <= current_price <= max_price and metrics['total_volume'] >= min_volume:
                    if metrics['price_change'] > price_change_threshold:
                        is_interesting = True
                        reasons.append(f"Price up {metrics['price_change']:.2f}%")

                    volume_increase = metrics['total_volume'] / metrics['avg_volume']
                    if volume_increase > volume_increase_threshold:
                        is_interesting = True
                        reasons.append(f"Volume up {volume_increase:.2f}x average")

                    if is_interesting:
                        alert_message = f"🚀 HIGH POTENTIAL ALERT for {metrics['symbol']}! 🚀\n"
                        alert_message += f"Reasons: {', '.join(reasons)}\n"
                        alert_message += f"Current Price: ${current_price:.2f}\n"
                        alert_message += f"Today's Volume: {metrics['total_volume']:.0f}\n"
                        alert_message += f"5-day Avg Volume: {metrics['avg_volume']:.0f}\n"
                        alert_message += f"Volatility (ATR): ${metrics['atr']:.2f}\n"
                        alert_message += f"5-day SMA: ${metrics['sma_5']:.2f}\n"
                        alert_message += f"20-day SMA: ${metrics['sma_20']:.2f}\n"

                        # Exit price suggestions
                        alert_message += f"Conservative Exit: ${metrics['conservative_exit']:.2f}\n"
                        alert_message += f"Moderate Exit: ${metrics['moderate_exit']:.2f}\n"
                        alert_message += f"Aggressive Exit: ${metrics['aggressive_exit']:.2f}\n"

                        # Options suggestions
                        alert_message += "\nOptions Considerations:\n"
                        alert_message += "1. Call options: Consider near-the-money calls if bullish\n"
                        alert_message += "2. Put options: Consider buying puts for downside protection\n"
                        alert_message += "3. Covered calls: If you own shares, selling calls can generate income\n"
                        alert_message += "4. Cash-secured puts: Sell puts if you're willing to buy at a lower price\n"

                        alert_message += "\nRemember to check option liquidity and implied volatility before trading!"

                        print(alert_message)
                        await self.send_discord_alert(alert_message)
                        await self.send_slack_alert(alert_message)

    async def monitor_all_stocks(self):
        self.session = aiohttp.ClientSession()
        try:
            # Process stocks in batches to avoid overwhelming the API
            batch_size = 10
            for i in range(0, len(self.symbols), batch_size):
                batch = self.symbols[i:i+batch_size]
                tasks = [self.monitor_stock(symbol) for symbol in batch]
                await asyncio.gather(*tasks)
                await asyncio.sleep(5)  # Wait 5 seconds between batches
        finally:
            await self.session.close()

    def remove_problematic_symbols(self):
        for symbol, count in self.error_count.items():
            if count > 3:  # Remove symbols that have failed more than 3 times
                if symbol in self.symbols:
                    self.symbols.remove(symbol)
                    print(f"Removed {symbol} due to repeated errors.")
        self.error_count.clear()

async def main():
    monitor = StockMonitor()
    monitor.symbols = await monitor.get_stock_symbols()
    print(f"Monitoring {len(monitor.symbols)} stocks for high-potential opportunities...")

    while True:
        await monitor.monitor_all_stocks()
        monitor.remove_problematic_symbols()
        print(f"Now monitoring {len(monitor.symbols)} stocks...")
        await asyncio.sleep(300)  # Wait for 5 minutes before next check

if __name__ == "__main__":
    asyncio.run(main())
