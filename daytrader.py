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

# Configuration settings
ACCOUNT_BALANCE = 100000  # Initial account balance
TRADING_INTERVAL = '5m'  # Time interval for stock data
TRADING_PERIOD = '1d'  # Period for stock data (changed to 1d)
CHECK_INTERVAL_MINUTES = 60  # How often to check open positions
SLEEP_MINUTES = 5  # Time to wait between iterations
MAX_STOCKS = 20  # Maximum number of stocks to analyze
RISK_FACTOR = 0.01  # Risk factor for position sizing
RSI_OVERSOLD = 30  # RSI level considered oversold
RSI_OVERBOUGHT = 70  # RSI level considered overbought
MINIMUM_DATA_POINTS = 30  # Minimum number of data points required for analysis (adjusted)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database setup
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

async def get_stock_data(session: aiohttp.ClientSession, ticker: str, interval: str, period: str) -> pd.DataFrame:
    """
    Asynchronously fetch stock data using yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        data = await asyncio.to_thread(stock.history, period=period, interval=interval)
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for the given stock data.
    """
    if len(data) < MINIMUM_DATA_POINTS:
        logger.warning(f"Insufficient data points ({len(data)}) for calculating indicators. Minimum required: {MINIMUM_DATA_POINTS}")
        return pd.DataFrame()

    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'], data['MACD_Signal'], _ = calculate_macd(data['Close'])
    return data

def calculate_rsi(close: pd.Series, timeperiod: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI).
    """
    diff = close.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    ema_up = up.ewm(com=timeperiod-1, adjust=False).mean()
    ema_down = down.ewm(com=timeperiod-1, adjust=False).mean()
    rs = ema_up / ema_down
    return 100 - (100 / (1 + rs))

def calculate_macd(close: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate the Moving Average Convergence Divergence (MACD).
    """
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def check_signals(data: pd.DataFrame, risk_factor: float) -> Tuple[bool, bool, float]:
    """
    Check for buy and sell signals based on technical indicators.
    """
    if data.empty or len(data) < 2:
        return False, False, 0

    try:
        buy_signal = (
            (data['Close'].iloc[-1] > data['SMA_20'].iloc[-1]) and
            (data['RSI'].iloc[-1] < RSI_OVERSOLD) and
            (data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1])
        )
        sell_signal = (
            (data['Close'].iloc[-1] < data['SMA_50'].iloc[-1]) and
            (data['RSI'].iloc[-1] > RSI_OVERBOUGHT) and
            (data['MACD'].iloc[-1] < data['MACD_Signal'].iloc[-1])
        )

        # Calculate position size based on risk
        atr = calculate_atr(data)
        position_size = calculate_position_size(data['Close'].iloc[-1], atr, risk_factor)

        return buy_signal, sell_signal, position_size
    except KeyError as e:
        logger.error(f"Missing key in data: {e}")
        return False, False, 0

def calculate_atr(data: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate the Average True Range (ATR) for position sizing.
    """
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean().iloc[-1]

def calculate_position_size(price: float, atr: float, risk_factor: float) -> float:
    """
    Calculate the position size based on the ATR and risk factor.
    """
    risk_per_trade = ACCOUNT_BALANCE * risk_factor
    stop_loss = 2 * atr  # Set stop loss at 2 * ATR
    shares = risk_per_trade / stop_loss
    return np.floor(shares)

async def get_most_active_stocks(session: aiohttp.ClientSession) -> List[str]:
    """
    Asynchronously fetch the most active stocks.
    """
    url = 'https://stockanalysis.com/markets/active/'
    try:
        async with session.get(url) as response:
            html = await response.text()
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table', {'id': 'main-table'})
            rows = table.find_all('tr')[1:]  # Exclude the header row
            symbols = [row.find_all('td')[1].text.strip() for row in rows]
            return symbols[:MAX_STOCKS]
    except Exception as e:
        logger.error(f"Error fetching active stocks: {str(e)}")
        return []

def calculate_trade_performance(buy_price: float, sell_price: float) -> Tuple[float, float]:
    """
    Calculate the profit/loss and percentage for a trade.
    """
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
                elif sell and ticker in trades:
                    buy_price = trades[ticker]['buy_price']
                    position_size = trades[ticker]['position_size']
                    profit_loss, profit_loss_percentage = calculate_trade_performance(buy_price, current_price)
                    total_pl = profit_loss * position_size
                    logger.info(f"SELL signal for {ticker} at price {current_price:.2f}")
                    logger.info(f"Trade performance for {ticker}: Profit/Loss = ${total_pl:.2f}, Profit/Loss % = {profit_loss_percentage:.2f}%")

                    # Record trade in database
                    cursor.execute('''
                        INSERT INTO trades (ticker, buy_time, buy_price, sell_time, sell_price, profit_loss, profit_loss_percentage)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (ticker, trades[ticker]['buy_time'], buy_price, datetime.now(), current_price, total_pl, profit_loss_percentage))
                    conn.commit()

                    del trades[ticker]

                # Check if any open trades have reached the check interval
                for ticker, trade in list(trades.items()):
                    if datetime.now() - trade['buy_time'] >= timedelta(minutes=CHECK_INTERVAL_MINUTES):
                        buy_price = trade['buy_price']
                        position_size = trade['position_size']
                        profit_loss, profit_loss_percentage = calculate_trade_performance(buy_price, current_price)
                        total_pl = profit_loss * position_size
                        logger.info(f"Closing trade for {ticker} after check interval. Profit/Loss = ${total_pl:.2f}, Profit/Loss % = {profit_loss_percentage:.2f}%")

                        # Record trade in database
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
