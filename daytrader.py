import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import logging
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sqlite3
import json

# Configuration settings
ACCOUNT_BALANCE = 100000
TRADING_INTERVAL = '5m'
TRADING_PERIOD = '1d'
CHECK_INTERVAL_MINUTES = 60
SLEEP_MINUTES = 5
MAX_STOCKS = 20
RISK_FACTOR = 0.01
RSI_OVERSOLD = 30  # Increased from 30
RSI_OVERBOUGHT = 70  # Decreased from 70
MINIMUM_DATA_POINTS = 30

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

conn = sqlite3.connect('trading_data.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY,
        ticker TEXT,
        buy_time TIMESTAMP,
        buy_price REAL,
        sell_time TIMESTAMP,
        sell_price REAL,
        profit_loss REAL,
        profit_loss_percentage REAL
    )
''')
conn.commit()

async def send_discord_alert(session: aiohttp.ClientSession, message: str):
    """Send an alert to Discord"""
    payload = {
        "content": message
    }
    try:
        async with session.post(DISCORD_WEBHOOK_URL, json=payload) as response:
            if response.status == 204:
                logger.info("Discord alert sent successfully")
            else:
                logger.error(f"Failed to send Discord alert. Status: {response.status}")
    except Exception as e:
        logger.error(f"Error sending Discord alert: {str(e)}")

async def get_stock_data(session: aiohttp.ClientSession, ticker: str, interval: str, period: str) -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker)
        data = await asyncio.to_thread(stock.history, period=period, interval=interval)
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    if len(data) < MINIMUM_DATA_POINTS:
        logger.warning(f"Insufficient data points ({len(data)}) for calculating indicators. Minimum required: {MINIMUM_DATA_POINTS}")
        return pd.DataFrame()

    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'], data['MACD_Signal'], _ = calculate_macd(data['Close'])
    return data

def calculate_rsi(close: pd.Series, timeperiod: int = 14) -> pd.Series:
    diff = close.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    ema_up = up.ewm(com=timeperiod-1, adjust=False).mean()
    ema_down = down.ewm(com=timeperiod-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def calculate_macd(close: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def check_signals(data: pd.DataFrame, risk_factor: float) -> Tuple[bool, bool, float]:
    if data.empty or len(data) < 2:
        return False, False, 0

    try:
        current_price = data['Close'].iloc[-1]
        current_rsi = data['RSI'].iloc[-1]
        current_macd = data['MACD'].iloc[-1]
        current_macd_signal = data['MACD_Signal'].iloc[-1]

        buy_signal = (
            (current_price > data['SMA_20'].iloc[-1]) and
            (current_rsi < RSI_OVERSOLD) and
            (current_macd > current_macd_signal)
        )
        sell_signal = (
            (current_price < data['SMA_50'].iloc[-1]) and
            (current_rsi > RSI_OVERBOUGHT) and
            (current_macd < current_macd_signal)
        )

        # Log the current values for debugging
        logger.info(f"Current price: {current_price:.2f}, SMA_20: {data['SMA_20'].iloc[-1]:.2f}, SMA_50: {data['SMA_50'].iloc[-1]:.2f}")
        logger.info(f"Current RSI: {current_rsi:.2f}, MACD: {current_macd:.2f}, MACD Signal: {current_macd_signal:.2f}")
        logger.info(f"Buy signal: {buy_signal}, Sell signal: {sell_signal}")

        atr = calculate_atr(data)
        position_size = calculate_position_size(current_price, atr, risk_factor)

        return buy_signal, sell_signal, position_size
    except KeyError as e:
        logger.error(f"Missing key in data: {e}")
        return False, False, 0

def calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean().iloc[-1]

def calculate_position_size(price: float, atr: float, risk_factor: float) -> float:
    risk_per_trade = ACCOUNT_BALANCE * risk_factor
    stop_loss = 2 * atr
    shares = risk_per_trade / stop_loss
    return np.floor(shares)

async def get_most_active_stocks(session: aiohttp.ClientSession) -> List[str]:
    url = 'https://stockanalysis.com/markets/gainers/'
    try:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table', {'id': 'main-table'})
            rows = table.find_all('tr')[1:]
            symbols = [row.find_all('td')[1].text.strip() for row in rows]
            return symbols[:MAX_STOCKS]
    except Exception as e:
        logger.error(f"Error fetching active stocks: {str(e)}")
        return []

def calculate_trade_performance(buy_price: float, sell_price: float) -> Tuple[float, float]:
    profit_loss = sell_price - buy_price
    profit_loss_percentage = (profit_loss / buy_price) * 100
    return profit_loss, profit_loss_percentage

async def main():
    trades: Dict[str, Dict] = {}

    async with aiohttp.ClientSession() as session:
        while True:
            logger.info("Fetching most active stocks...")
            tickers = await get_most_active_stocks(session)
            logger.info(f"Most Active Stocks: {', '.join(tickers)}")

            tasks = [get_stock_data(session, ticker, TRADING_INTERVAL, TRADING_PERIOD) for ticker in tickers]
            stock_data = await asyncio.gather(*tasks)

            for ticker, data in zip(tickers, stock_data):
                if data.empty:
                    logger.warning(f"No data available for {ticker}")
                    continue

                logger.info(f"Analyzing {ticker} with {len(data)} data points")
                data = calculate_indicators(data)
                if data.empty:
                    logger.warning(f"Insufficient data for calculating indicators for {ticker}")
                    continue

                buy, sell, position_size = check_signals(data, RISK_FACTOR)
                current_price = data['Close'].iloc[-1]

                if buy and ticker not in trades:
                    trades[ticker] = {'buy_time': datetime.now(), 'buy_price': current_price, 'position_size': position_size}
                    logger.info(f"BUY signal for {ticker} at price {current_price:.2f}, position size: {position_size}")
                    await send_discord_alert(session, f"BUY signal for {ticker} at price ${current_price:.2f}, position size: {position_size}")
                elif sell and ticker in trades:
                    buy_price = trades[ticker]['buy_price']
                    position_size = trades[ticker]['position_size']
                    profit_loss, profit_loss_percentage = calculate_trade_performance(buy_price, current_price)
                    total_pl = profit_loss * position_size
                    logger.info(f"SELL signal for {ticker} at price {current_price:.2f}")
                    logger.info(f"Trade performance for {ticker}: Profit/Loss = ${total_pl:.2f}, Profit/Loss % = {profit_loss_percentage:.2f}%")
                    
                    await send_discord_alert(session, f"SELL signal for {ticker} at price ${current_price:.2f}\n"
                                             f"Trade performance: Profit/Loss = ${total_pl:.2f}, Profit/Loss % = {profit_loss_percentage:.2f}%")

                    cursor.execute('''
                        INSERT INTO trades (ticker, buy_time, buy_price, sell_time, sell_price, profit_loss, profit_loss_percentage)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (ticker, trades[ticker]['buy_time'], buy_price, datetime.now(), current_price, total_pl, profit_loss_percentage))
                    conn.commit()

                    del trades[ticker]

                for ticker, trade in list(trades.items()):
                    if datetime.now() - trade['buy_time'] >= timedelta(minutes=CHECK_INTERVAL_MINUTES):
                        buy_price = trade['buy_price']
                        position_size = trade['position_size']
                        profit_loss, profit_loss_percentage = calculate_trade_performance(buy_price, current_price)
                        total_pl = profit_loss * position_size
                        logger.info(f"Closing trade for {ticker} after check interval. Profit/Loss = ${total_pl:.2f}, Profit/Loss % = {profit_loss_percentage:.2f}%")
                        
                        await send_discord_alert(session, f"Closing trade for {ticker} after check interval.\n"
                                                 f"Profit/Loss = ${total_pl:.2f}, Profit/Loss % = {profit_loss_percentage:.2f}%")

                        cursor.execute('''
                            INSERT INTO trades (ticker, buy_time, buy_price, sell_time, sell_price, profit_loss, profit_loss_percentage)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (ticker, trade['buy_time'], buy_price, datetime.now(), current_price, total_pl, profit_loss_percentage))
                        conn.commit()

                        del trades[ticker]

            logger.info(f"Sleeping for {SLEEP_MINUTES} minutes before next update...")
            await asyncio.sleep(SLEEP_MINUTES * 60)

if __name__ == "__main__":
    asyncio.run(main())
