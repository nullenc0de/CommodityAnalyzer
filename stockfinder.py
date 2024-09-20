"""
Stock Monitor

This script monitors stocks listed on NASDAQ, analyzes their performance,
and identifies high-potential stocks based on various technical indicators.
It uses the Yahoo Finance API to fetch stock data and can send notifications
to Discord and Slack when high-potential stocks are identified.

Main components:
1. StockMonitor class: Handles all stock monitoring and analysis operations.
2. get_stock_symbols: Fetches the list of NASDAQ-listed stocks.
3. get_stock_data: Retrieves historical stock data from Yahoo Finance.
4. calculate_metrics: Computes various technical indicators for a stock.
5. score_stock: Assigns a score to a stock based on its metrics.
6. monitor_stocks: Main loop that continuously monitors stocks.
7. send_discord_notification: Sends alerts to Discord.
8. send_slack_notification: Sends alerts to Slack.

Usage:
Set the DISCORD_WEBHOOK and SLACK_WEBHOOK environment variables before running.
Run the script with: python stock_monitor.py
"""

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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        logging.info("StockMonitor initialized")

    async def get_stock_symbols(self):
        logging.info("Fetching stock symbols...")
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
                        logging.info(f"Successfully fetched {self.total_symbols} symbols")
                        return symbols
                    else:
                        logging.error(f"Failed to fetch stock symbols: HTTP {response.status}")
                        return []
        except Exception as e:
            logging.error(f"Error in get_stock_symbols: {str(e)}")
            traceback.print_exc()
            return []

    async def get_stock_data(self, symbol):
        logging.info(f"Fetching data for {symbol}")
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
                        logging.warning(f"Stock {symbol} not found on Yahoo Finance")
                        return None
                    elif response.status == 429:
                        wait_time = 2 ** attempt + random.random()
                        logging.warning(f"Rate limit hit for {symbol}. Retrying in {wait_time:.2f} seconds...")
                        await asyncio.sleep(wait_time)
                    else:
                        logging.error(f"Error fetching data for {symbol}: HTTP Status {response.status}")
            except Exception as e:
                logging.error(f"Error fetching data for {symbol}: {str(e)}")

            if attempt < max_retries - 1:
                await asyncio.sleep(1)  # Wait 1 second before retrying

        return None

    def process_stock_data(self, symbol, data):
        logging.info(f"Processing data for {symbol}")
        if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
            result = data['chart']['result'][0]
            timestamps = result.get('timestamp')
            indicators = result.get('indicators', {}).get('quote', [{}])[0]

            if not timestamps or not indicators:
                logging.warning(f"Warning: Incomplete data received for {symbol}")
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
                logging.warning(f"Warning: No valid data points for {symbol}")
        else:
            logging.error(f"Error: Unexpected data format received for {symbol}")
        return None

    def calculate_metrics(self, stock_data):
        logging.info(f"Calculating metrics for {stock_data['symbol']}")
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
        bb_width = (upper[-1] - lower[-1]) / middle[-1] if middle[-1] != 0 else None

        recent_high = df['High'].tail(20).max()
        recent_low = df['Low'].tail(20).min()

        conservative_exit = min(current_price * 1.05, current_price + atr) if atr is not None else current_price * 1.05
        moderate_exit = min(current_price * 1.10, current_price + 1.5 * atr) if atr is not None else current_price * 1.10
        aggressive_exit = min(current_price * 1.15, recent_high)

        if current_price < sma_20:
            conservative_exit = min(conservative_exit, sma_20)
        if current_price < sma_5:
            moderate_exit = max(moderate_exit, sma_5)

        potential_profit = (aggressive_exit - current_price) / current_price * 100

        # Risk management calculations
        stop_loss = current_price - 2 * atr if atr is not None else current_price * 0.95
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
            'risk_reward_ratio': risk_reward_ratio,
            'stop_loss': stop_loss
        }

    def score_stock(self, metrics):
        logging.info(f"Scoring stock {metrics['symbol']}")
        score = 0
        reasons = []

        # Price
        if metrics['current_price'] < 5:
            score -= 1
            reasons.append(f"Low current price: {metrics['current_price']:.2f}")
        elif metrics['current_price'] > 100:
            score -= 1
            reasons.append(f"High current price: {metrics['current_price']:.2f}")

        # Volume
        if metrics['total_volume'] > 1000000:
            score += 1
        else:
            score -= 1
            reasons.append(f"Low total volume: {metrics['total_volume']:.0f}")

        # Volatility
        if metrics['volatility'] is not None and metrics['volatility'] > 0.05:
            score += 1
        else:
            score -= 1
            reasons.append(f"Low volatility: {metrics['volatility']:.4f}")

        # RSI
        if metrics['rsi'] < 30:
            score += 2
            reasons.append(f"RSI indicates oversold: {metrics['rsi']:.2f}")
        elif metrics['rsi'] > 70:
            score -= 1
            reasons.append(f"RSI indicates overbought: {metrics['rsi']:.2f}")

        # MACD
        if metrics['macd'] > metrics['macd_signal']:
            score += 1
        else:
            score -= 1
            reasons.append(f"MACD is below signal: MACD={metrics['macd']:.4f}, Signal={metrics['macd_signal']:.4f}")

        # Bollinger Bands
        if metrics['bb_width'] is not None and metrics['bb_width'] < 0.1:
            score += 1
        else:
            score -= 1
            reasons.append(f"Wide Bollinger Bands: {metrics['bb_width']:.4f}")

        # Risk/Reward Ratio
        if metrics['risk_reward_ratio'] > 2:
            score += 2
        elif metrics['risk_reward_ratio'] < 1:
            score -= 2
            reasons.append(f"Risk/Reward ratio is unfavorable: {metrics['risk_reward_ratio']:.2f}")

        # Potential Profit
        if metrics['potential_profit'] > 10:
            score += 2
            reasons.append(f"High potential profit: {metrics['potential_profit']:.2f}%")

        return score, reasons

    async def monitor_stocks(self):
        logging.info("Starting stock monitoring...")
        self.symbols = await self.get_stock_symbols()
        if not self.symbols:
            logging.warning("No stock symbols to monitor.")
            return

        async with aiohttp.ClientSession() as session:
            self.session = session
            self.is_monitoring = True
            while self.is_monitoring:
                try:
                    logging.info("Starting new monitoring cycle...")
                    random.shuffle(self.symbols)

                    for symbol in self.symbols:
                        stock_data = await self.get_stock_data(symbol)
                        if stock_data:
                            metrics = self.calculate_metrics(stock_data)
                            if metrics:
                                score, reasons = self.score_stock(metrics)
                                if score > 5:  # Threshold for high-potential stocks
                                    analysis = {
                                        'symbol': symbol,
                                        'score': score,
                                        'reasons': reasons,
                                        'metrics': metrics
                                    }
                                    self.high_potential_stocks.append(analysis)
                                    logging.info(f"High potential stock detected: {symbol}, Score: {score}")
                                    await self.send_discord_notification(analysis)
                                    await self.send_slack_notification(analysis)

                        logging.info(f"Processed {symbol}")

                    logging.info(f"Completed monitoring cycle with {len(self.high_potential_stocks)} high-potential stocks.")

                    logging.info(f"Sleeping for {self.monitor_interval} seconds before next cycle...")
                    await asyncio.sleep(self.monitor_interval)

                except Exception as e:
                    logging.error(f"Error in monitoring loop: {str(e)}")
                    traceback.print_exc()

            logging.info("Stock monitoring stopped.")

    async def send_discord_notification(self, analysis):
        logging.info(f"Sending Discord notification for {analysis['symbol']}")
        if not self.discord_webhook:
            logging.warning("Discord webhook URL is not set. Skipping notification.")
            return

        try:
            async with aiohttp.ClientSession() as session:
                webhook = Webhook.from_url(self.discord_webhook, session=session)
                embed = {
                    "title": f"High Potential Stock: {analysis['symbol']}",
                    "description": f"Score: {analysis['score']}\nReasons: {', '.join(analysis['reasons'])}",
                    "fields": [
                        {"name": "Current Price", "value": f"${analysis['metrics']['current_price']:.2f}", "inline": True},
                        {"name": "Potential Profit", "value": f"{analysis['metrics']['potential_profit']:.2f}%", "inline": True},
                        {"name": "Risk-Reward Ratio", "value": f"{analysis['metrics']['risk_reward_ratio']:.2f}", "inline": True},
                        {"name": "Conservative Exit", "value": f"${analysis['metrics']['conservative_exit']:.2f}", "inline": True},
                        {"name": "Moderate Exit", "value": f"${analysis['metrics']['moderate_exit']:.2f}", "inline": True},
                        {"name": "Aggressive Exit", "value": f"${analysis['metrics']['aggressive_exit']:.2f}", "inline": True},
                        {"name": "Stop Loss", "value": f"${analysis['metrics']['stop_loss']:.2f}", "inline": True},
                        {"name": "RSI", "value": f"{analysis['metrics']['rsi']:.2f}", "inline": True},
                        {"name": "MACD", "value": f"{analysis['metrics']['macd']:.4f}", "inline": True}
                    ],
                    "color": 0x00ff00  # Green color
                }
                await webhook.send(embed=embed)
                logging.info(f"Discord notification sent for {analysis['symbol']}")
        except Exception as e:
            logging.error(f"Error sending Discord notification: {str(e)}")
            traceback.print_exc()

    async def send_slack_notification(self, analysis):
        logging.info(f"Sending Slack notification for {analysis['symbol']}")
        if not self.slack_webhook:
            logging.warning("Slack webhook URL is not set. Skipping notification.")
            return

        try:
            message = {
                "text": f"High Potential Stock: {analysis['symbol']}",
                "attachments": [
                    {
                        "color": "#36a64f",
                        "fields": [
                            {"title": "Score", "value": str(analysis['score']), "short": True},
                            {"title": "Current Price", "value": f"${analysis['metrics']['current_price']:.2f}", "short": True},
                            {"title": "Potential Profit", "value": f"{analysis['metrics']['potential_profit']:.2f}%", "short": True},
                            {"title": "Risk-Reward Ratio", "value": f"{analysis['metrics']['risk_reward_ratio']:.2f}", "short": True},
                            {"title": "Conservative Exit", "value": f"${analysis['metrics']['conservative_exit']:.2f}", "short": True},
                            {"title": "Moderate Exit", "value": f"${analysis['metrics']['moderate_exit']:.2f}", "short": True},
                            {"title": "Aggressive Exit", "value": f"${analysis['metrics']['aggressive_exit']:.2f}", "short": True},
                            {"title": "Stop Loss", "value": f"${analysis['metrics']['stop_loss']:.2f}", "short": True},
                            {"title": "RSI", "value": f"{analysis['metrics']['rsi']:.2f}", "short": True},
                            {"title": "MACD", "value": f"{analysis['metrics']['macd']:.4f}", "short": True},
                            {"title": "Reasons", "value": ", ".join(analysis['reasons'])}
                        ]
                    }
                ]
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(self.slack_webhook, json=message) as response:
                    if response.status == 200:
                        logging.info(f"Slack notification sent for {analysis['symbol']}")
                    else:
                        logging.error(f"Failed to send Slack notification: HTTP {response.status}")
        except Exception as e:
            logging.error(f"Error sending Slack notification: {str(e)}")
            traceback.print_exc()

# Main execution
if __name__ == "__main__":
    logging.info("Script started")
    monitor = StockMonitor()
    try:
        asyncio.run(monitor.monitor_stocks())
    except KeyboardInterrupt:
        logging.info("\nMonitoring stopped by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        traceback.print_exc()
    finally:
        logging.info("Stock monitoring has been stopped and cleaned up.")

    # Print summary of high potential stocks
    if monitor.high_potential_stocks:
        logging.info("\nHigh Potential Stocks Summary:")
        for stock in monitor.high_potential_stocks:
            logging.info(f"\nSymbol: {stock['symbol']}")
            logging.info(f"Score: {stock['score']}")
            logging.info(f"Current Price: ${stock['metrics']['current_price']:.2f}")
            logging.info(f"Potential Profit: {stock['metrics']['potential_profit']:.2f}%")
            logging.info(f"Risk-Reward Ratio: {stock['metrics']['risk_reward_ratio']:.2f}")
            logging.info(f"Conservative Exit: ${stock['metrics']['conservative_exit']:.2f}")
            logging.info(f"Moderate Exit: ${stock['metrics']['moderate_exit']:.2f}")
            logging.info(f"Aggressive Exit: ${stock['metrics']['aggressive_exit']:.2f}")
            logging.info(f"Stop Loss: ${stock['metrics']['stop_loss']:.2f}")
            logging.info("Reasons:")
            for reason in stock['reasons']:
                logging.info(f"- {reason}")
    else:
        logging.info("\nNo high potential stocks were found during this run.")

    logging.info("Script execution completed.")
