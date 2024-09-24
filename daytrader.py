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
import backoff
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# Configuration settings
ACCOUNT_BALANCE = 100000
TRADING_INTERVAL = '1h'
TRADING_PERIOD = '1y'  # Extended for backtesting
CHECK_INTERVAL_MINUTES = 60
SLEEP_MINUTES = 5
MAX_STOCKS = 20
INITIAL_RISK_FACTOR = 0.01
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MINIMUM_DATA_POINTS = 100
PAPER_TRADING = True
BACKTESTING = True  # Set to True to run backtesting

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1286420702173597807/hNgcuYY68fm6t0ncWSSGt2QwrQvEybW5uRrr2nXZCMiizQnq6Wguhm41SBJcO8TicQWy"

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
        profit_loss_percentage REAL,
        is_paper_trade BOOLEAN
    )
''')
conn.commit()

async def send_discord_alert(session: aiohttp.ClientSession, message: str):
    payload = {"content": message}
    try:
        async with session.post(DISCORD_WEBHOOK_URL, json=payload) as response:
            if response.status == 204:
                logger.info("Discord alert sent successfully")
            else:
                logger.error(f"Failed to send Discord alert. Status: {response.status}")
    except Exception as e:
        logger.error(f"Error sending Discord alert: {str(e)}")

@backoff.on_exception(backoff.expo, Exception, max_tries=3)
async def get_stock_data(session: aiohttp.ClientSession, ticker: str, interval: str, period: str) -> pd.DataFrame:
    try:
        stock = yf.Ticker(ticker)
        data = await asyncio.to_thread(stock.history, period=period, interval=interval)
        if len(data) < MINIMUM_DATA_POINTS:
            logger.warning(f"Insufficient data points for {ticker}: {len(data)}. Minimum required: {MINIMUM_DATA_POINTS}")
            return pd.DataFrame()
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {str(e)}")
        return pd.DataFrame()

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    # Trend Indicators
    data['SMA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
    data['EMA_50'] = EMAIndicator(close=data['Close'], window=50).ema_indicator()
    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()

    # Momentum Indicators
    data['RSI'] = RSIIndicator(close=data['Close']).rsi()
    stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()

    # Volatility Indicators
    bb = BollingerBands(close=data['Close'])
    data['BB_Upper'] = bb.bollinger_hband()
    data['BB_Lower'] = bb.bollinger_lband()
    data['ATR'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()

    return data

def check_signals(data: pd.DataFrame) -> Tuple[bool, bool, float]:
    if data.empty or len(data) < 2:
        return False, False, 0

    try:
        current_price = data['Close'].iloc[-1]
        current_rsi = data['RSI'].iloc[-1]
        current_macd = data['MACD'].iloc[-1]
        current_macd_signal = data['MACD_Signal'].iloc[-1]
        current_stoch_k = data['Stoch_K'].iloc[-1]
        current_stoch_d = data['Stoch_D'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        ema_50 = data['EMA_50'].iloc[-1]
        bb_upper = data['BB_Upper'].iloc[-1]
        bb_lower = data['BB_Lower'].iloc[-1]
        atr = data['ATR'].iloc[-1]

        buy_signal = (
            (current_price > sma_20) and
            (current_price > ema_50) and
            (current_rsi < RSI_OVERSOLD) and
            (current_macd > current_macd_signal) and
            (current_stoch_k > current_stoch_d) and
            (current_stoch_k < 20) and
            (current_price < bb_lower)
        )
        sell_signal = (
            (current_price < sma_20) and
            (current_price < ema_50) and
            (current_rsi > RSI_OVERBOUGHT) and
            (current_macd < current_macd_signal) and
            (current_stoch_k < current_stoch_d) and
            (current_stoch_k > 80) and
            (current_price > bb_upper)
        )

        # Dynamic risk factor based on volatility
        volatility = atr / current_price
        risk_factor = INITIAL_RISK_FACTOR * (1 + volatility)

        position_size = calculate_position_size(current_price, atr, risk_factor)

        logger.info(f"Current price: {current_price:.2f}, SMA_20: {sma_20:.2f}, EMA_50: {ema_50:.2f}")
        logger.info(f"Current RSI: {current_rsi:.2f}, MACD: {current_macd:.2f}, MACD Signal: {current_macd_signal:.2f}")
        logger.info(f"Stoch K: {current_stoch_k:.2f}, Stoch D: {current_stoch_d:.2f}")
        logger.info(f"BB Upper: {bb_upper:.2f}, BB Lower: {bb_lower:.2f}")
        logger.info(f"Buy signal: {buy_signal}, Sell signal: {sell_signal}, Risk Factor: {risk_factor:.4f}")

        return buy_signal, sell_signal, position_size
    except KeyError as e:
        logger.error(f"Missing key in data: {e}")
        return False, False, 0

def calculate_position_size(price: float, atr: float, risk_factor: float) -> float:
    risk_per_trade = ACCOUNT_BALANCE * risk_factor
    stop_loss = 2 * atr
    shares = risk_per_trade / stop_loss
    return np.floor(shares)

async def get_most_active_stocks(session: aiohttp.ClientSession) -> List[str]:
    url = 'https://stockanalysis.com/markets/active/'
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

def check_exit_conditions(data: pd.DataFrame, entry_price: float, position_size: float) -> Tuple[bool, str]:
    current_price = data['Close'].iloc[-1]
    atr = data['ATR'].iloc[-1]
    
    # Trailing stop loss (2 * ATR)
    trailing_stop = entry_price - (2 * atr)
    if current_price <= trailing_stop:
        return True, "Trailing Stop Loss"
    
    # Take profit (3 * ATR)
    take_profit = entry_price + (3 * atr)
    if current_price >= take_profit:
        return True, "Take Profit"
    
    # Time-based exit (after 5 days)
    days_held = (data.index[-1] - data.index[0]).days
    if days_held >= 5:
        return True, "Time-based Exit"
    
    return False, ""

def backtest(data: pd.DataFrame) -> Dict[str, float]:
    balance = ACCOUNT_BALANCE
    trades = []
    position = None

    for i in range(len(data)):
        if i < MINIMUM_DATA_POINTS:
            continue

        current_data = data.iloc[:i+1]
        buy, sell, position_size = check_signals(current_data)

        if position is None and buy:
            position = {
                'entry_price': current_data['Close'].iloc[-1],
                'size': position_size,
                'entry_date': current_data.index[-1]
            }
        elif position is not None:
            exit_condition, _ = check_exit_conditions(current_data, position['entry_price'], position['size'])
            if sell or exit_condition:
                exit_price = current_data['Close'].iloc[-1]
                profit_loss, _ = calculate_trade_performance(position['entry_price'], exit_price)
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': current_data.index[-1],
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'profit_loss': profit_loss * position['size']
                })
                balance += trades[-1]['profit_loss']
                position = None

    total_trades = len(trades)
    winning_trades = sum(1 for trade in trades if trade['profit_loss'] > 0)
    total_profit = sum(trade['profit_loss'] for trade in trades)
    max_drawdown = min(trade['profit_loss'] for trade in trades) if trades else 0

    return {
        'final_balance': balance,
        'total_profit': total_profit,
        'total_trades': total_trades,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        'max_drawdown': max_drawdown
    }

async def main():
    trades: Dict[str, Dict] = {}

    async with aiohttp.ClientSession() as session:
        if BACKTESTING:
            logger.info("Starting backtesting...")
            tickers = await get_most_active_stocks(session)
            for ticker in tickers:
                data = await get_stock_data(session, ticker, TRADING_INTERVAL, TRADING_PERIOD)
                if not data.empty:
                    data = calculate_indicators(data)
                    results = backtest(data)
                    logger.info(f"Backtesting results for {ticker}:")
                    logger.info(f"Final Balance: ${results['final_balance']:.2f}")
                    logger.info(f"Total Profit: ${results['total_profit']:.2f}")
                    logger.info(f"Total Trades: {results['total_trades']}")
                    logger.info(f"Win Rate: {results['win_rate']:.2%}")
                    logger.info(f"Max Drawdown: ${results['max_drawdown']:.2f}")
            logger.info("Backtesting completed.")
            return

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

                buy, sell, position_size = check_signals(data)
                current_price = data['Close'].iloc[-1]

                if buy and ticker not in trades:
                    trades[ticker] = {'buy_time': datetime.now(), 'buy_price': current_price, 'position_size': position_size}
                    logger.info(f"{'PAPER ' if PAPER_TRADING else ''}BUY signal for {ticker} at price {current_price:.2f}, position size: {position_size}")
                    await send_discord_alert(session, f"{'PAPER ' if PAPER_TRADING else ''}BUY signal for {ticker} at price ${current_price:.2f}, position size: {position_size}")
                
                elif ticker in trades:
                    exit_condition, exit_reason = check_exit_conditions(data, trades[ticker]['buy_price'], trades[ticker]['position_size'])
                    if sell or exit_condition:
                        buy_price = trades[ticker]['buy_price']
                        position_size = trades[ticker]['position_size']
                        profit_loss, profit_loss_percentage = calculate_trade_performance(buy_price, current_price)
                        total_pl = profit_loss * position_size
                        logger.info(f"{'PAPER ' if PAPER_TRADING else ''}SELL signal for {ticker} at price {current_price:.2f}")
                        logger.info(f"Trade performance for {ticker}: Profit/Loss = ${total_pl:.2f}, Profit/Loss % = {profit_loss_percentage:.2f}%")
                        logger.info(f"Exit reason: {exit_reason if exit_condition else 'Sell Signal'}")
                        
                        await send_discord_alert(session, f"{'PAPER ' if PAPER_TRADING else ''}SELL signal for {ticker} at price ${current_price:.2f}\n"
                                                 f"Trade performance: Profit/Loss = ${total_pl:.2f}, Profit/Loss % = {profit_loss_percentage:.2f}%\n"
                                                 f"Exit reason: {exit_reason if exit_condition else 'Sell Signal'}")

                        cursor.execute('''
                            INSERT INTO trades (ticker, buy_time, buy_price, sell_time, sell_price, profit_loss, profit_loss_percentage, is_paper_trade)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (ticker, trades[ticker]['buy_time'], buy_price, datetime.now(), current_price, total_pl, profit_loss_percentage, PAPER_TRADING))
                        conn.commit()

                        del trades[ticker]

            logger.info(f"Sleeping for {SLEEP_MINUTES} minutes before next update...")
            await asyncio.sleep(SLEEP_MINUTES * 60)

if __name__ == "__main__":
    asyncio.run(main())
