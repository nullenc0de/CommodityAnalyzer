import yfinance as yf
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import sqlite3
import json
import backoff
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from textblob import TextBlob
import requests
import random

# Configuration settings
ACCOUNT_BALANCE = 100000
TRADING_INTERVAL = '1d'
TRADING_PERIOD = '1y'
CHECK_INTERVAL_MINUTES = 60
SLEEP_MINUTES = 5
MAX_STOCKS = 20
INITIAL_RISK_FACTOR = 0.01
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MINIMUM_DATA_POINTS = 100
PAPER_TRADING = True
BACKTESTING = True  # Set to True for backtesting, False for live trading

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/your_webhook_url_here"
NEWS_API_KEY = "your_news_api_key_here"

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
        is_paper_trade INTEGER
    )
''')
conn.commit()

POPULAR_STOCKS = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corporation
    "AMZN",  # Amazon.com, Inc.
    "GOOGL",  # Alphabet Inc. (Google)
    "FB",  # Meta Platforms, Inc.
    "TSLA",  # Tesla, Inc.
    "BRK-B",  # Berkshire Hathaway Inc.
    "JNJ",  # Johnson & Johnson
    "NVDA",  # NVIDIA Corporation
    "PYPL",  # PayPal Holdings, Inc.
    "ADBE",  # Adobe Inc.
    "CRM",  # Salesforce, Inc.
    "NFLX",  # Netflix, Inc.
    "INTC",  # Intel Corporation
    "AMD",  # Advanced Micro Devices, Inc.
    "TXN",  # Texas Instruments Incorporated
    "VRTX",  # Vertex Pharmaceuticals Incorporated
    "ZM",  # Zoom Video Communications, Inc.
    "BABA",  # Alibaba Group Holding Limited
    "BABA",  # Alibaba Group Holding Limited
    "TCEHY",  # Tencent Holdings Limited
    "SHOP",  # Shopify Inc.
    "COIN",  # Coinbase Global, Inc.
    "TWTR",  # Twitter, Inc.
    "SNAP",  # Snap Inc.
    "UBER",  # Uber Technologies, Inc.
    "LYFT",  # Lyft, Inc.
    "SQ",  # Square Inc.
    "ROKU",  # Roku, Inc.
    "ETSY",  # Etsy, Inc.
    "DOCU",  # DocuSign, Inc.
    "WDAY",  # Workday, Inc.
    "OKTA",  # Okta, Inc.
    "TEAM",  # Atlassian Corporation Plc
    "HUBS",  # HubSpot, Inc.
    "ASML",  # ASML Holding N.V.
    "LRCN",  # Lear Corporation
    "LMT",  # Lockheed Martin Corporation
    "BA",  # Boeing Co.
    "UNP",  # Union Pacific Corporation
    "CSX",  # CSX Corporation
    "KHC",  # Kraft Heinz Company
    "PG",  # Procter & Gamble Company
    "KO",  # The Coca-Cola Company
    "PEP",  # PepsiCo, Inc.
    "JPM",  # JPMorgan Chase & Co.
    "BAC",  # Bank of America Corporation
    "WFC",  # Wells Fargo & Company
    "C",  # Citigroup Inc.
    "GS",  # The Goldman Sachs Group, Inc.
    "V",  # Visa Inc.
    "MA",  # Mastercard Incorporated
    "ADSK",  # Autodesk, Inc.
    "INTU",  # Intuit Inc.
    "ORCL",  # Oracle Corporation
    "SAP",  # SAP SE
    "MU",  # Micron Technology, Inc.
    "QCOM",  # Qualcomm Incorporated
    "AVGO",  # Broadcom Inc.
    "TXN",  # Texas Instruments Incorporated
    "LRCN",  # Lear Corporation
    "LMT",  # Lockheed Martin Corporation
    "BA",  # Boeing Co.
    "UNP",  # Union Pacific Corporation
    "CSX",  # CSX Corporation
    "KHC",  # Kraft Heinz Company
    "PG",  # Procter & Gamble Company
    "KO",  # The Coca-Cola Company
    "PEP",  # PepsiCo, Inc.
    "JPM",  # JPMorgan Chase & Co.
    "BAC",  # Bank of America Corporation
    "WFC",  # Wells Fargo & Company
    "C",  # Citigroup Inc.
    "GS",  # The Goldman Sachs Group, Inc.
    "V",  # Visa Inc.
    "MA",  # Mastercard Incorporated
    "ADSK",  # Autodesk, Inc.
    "INTU",  # Intuit Inc.
    "ORCL",  # Oracle Corporation
    "SAP",  # SAP SE
    "MU",  # Micron Technology, Inc.
    "QCOM",  # Qualcomm Incorporated
    "AVGO",  # Broadcom Inc.
]

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
    data['SMA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
    data['EMA_50'] = EMAIndicator(close=data['Close'], window=50).ema_indicator()
    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    data['RSI'] = RSIIndicator(close=data['Close']).rsi()
    stoch = StochasticOscillator(high=data['High'], low=data['Low'], close=data['Close'])
    data['Stoch_K'] = stoch.stoch()
    data['Stoch_D'] = stoch.stoch_signal()
    bb = BollingerBands(close=data['Close'])
    data['BB_Upper'] = bb.bollinger_hband()
    data['BB_Lower'] = bb.bollinger_lband()
    data['ATR'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
    return data

def prepare_data_for_ml(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    features = ['SMA_20', 'EMA_50', 'MACD', 'MACD_Signal', 'RSI', 'Stoch_K', 'Stoch_D', 'BB_Upper', 'BB_Lower', 'ATR',
                'Volume', 'Volume_SMA']
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Drop rows with NaN values
    data_cleaned = data.dropna()
    
    X = data_cleaned[features]
    y = data_cleaned['Target']
    
    return X, y

def train_ml_model(data: pd.DataFrame):
    X, y = prepare_data_for_ml(data)
    
    if len(X) < 100:  # Arbitrary threshold, adjust as needed
        logger.warning(f"Insufficient data for ML model training. Data points: {len(X)}")
        return None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    logger.info(f"Model accuracy: {accuracy:.2f}")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    return best_model

def ai_trade_decision(model, current_data: pd.DataFrame) -> bool:
    if model is None:
        return True
    features = ['SMA_20', 'EMA_50', 'MACD', 'MACD_Signal', 'RSI', 'Stoch_K', 'Stoch_D', 'BB_Upper', 'BB_Lower', 'ATR',
                'Volume', 'Volume_SMA']
    prediction = model.predict_proba(current_data[features].iloc[-1].to_frame().T)[0][1]
    return prediction > 0.6

def check_signals(data: pd.DataFrame, model, ticker: str) -> Tuple[bool, bool, float]:
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
        volume = data['Volume'].iloc[-1]
        avg_volume = data['Volume_SMA'].iloc[-1]

        ai_decision = ai_trade_decision(model, data)
        sentiment = get_news_sentiment(ticker)

        # Relaxed buy conditions
        buy_signal = (
            (current_price > sma_20) and
            (current_macd > current_macd_signal) and
            (current_stoch_k > current_stoch_d) and
            (current_rsi < 70) and  # Changed from 60 to 70
            (volume > 0.8 * avg_volume) and  # Relaxed volume condition
            ai_decision
        )

        # Relaxed sell conditions
        sell_signal = (
            (current_price < sma_20) and
            (current_macd < current_macd_signal) and
            (current_stoch_k < current_stoch_d) and
            (current_rsi > 30) and  # Changed from 40 to 30
            (volume > 0.8 * avg_volume) and  # Relaxed volume condition
            not ai_decision
        )

        volatility = atr / current_price
        risk_factor = INITIAL_RISK_FACTOR * (1 + volatility)

        position_size = calculate_position_size(current_price, atr, risk_factor)

        logger.info(f"Current price: {current_price:.2f}, SMA_20: {sma_20:.2f}, EMA_50: {ema_50:.2f}")
        logger.info(f"Current RSI: {current_rsi:.2f}, MACD: {current_macd:.2f}, MACD Signal: {current_macd_signal:.2f}")
        logger.info(f"Stoch K: {current_stoch_k:.2f}, Stoch D: {current_stoch_d:.2f}")
        logger.info(f"BB Upper: {bb_upper:.2f}, BB Lower: {bb_lower:.2f}")
        logger.info(f"Volume: {volume:.0f}, Avg Volume: {avg_volume:.0f}")
        logger.info(f"News Sentiment: {sentiment:.2f}")
        logger.info(f"Buy signal: {buy_signal}, Sell signal: {sell_signal}, Risk Factor: {risk_factor:.4f}")

        return buy_signal, sell_signal, position_size
    except KeyError as e:
        logger.error(f"Missing key in data: {e}")
        return False, False, 0

def calculate_position_size(price: float, atr: float, risk_factor: float) -> float:
    risk_per_trade = ACCOUNT_BALANCE * risk_factor
    stop_loss = 1.5 * atr
    shares = risk_per_trade / stop_loss
    return np.floor(shares)

async def get_most_active_stocks(session: aiohttp.ClientSession) -> List[str]:
    return random.sample(POPULAR_STOCKS, min(MAX_STOCKS, len(POPULAR_STOCKS)))

async def get_gap_up_stocks(session: aiohttp.ClientSession) -> List[str]:
    return random.sample(POPULAR_STOCKS, min(MAX_STOCKS // 2, len(POPULAR_STOCKS)))

async def get_combined_stock_list(session: aiohttp.ClientSession) -> List[str]:
    most_active = await get_most_active_stocks(session)
    gap_up = await get_gap_up_stocks(session)
    combined = list(set(most_active + gap_up))
    return combined[:MAX_STOCKS]

def calculate_trade_performance(buy_price: float, sell_price: float) -> Tuple[float, float]:
    profit_loss = sell_price - buy_price
    profit_loss_percentage = (profit_loss / buy_price) * 100
    return profit_loss, profit_loss_percentage

def check_exit_conditions(data: pd.DataFrame, entry_price: float, position_size: float) -> Tuple[bool, str]:
    current_price = data['Close'].iloc[-1]
    atr = data['ATR'].iloc[-1]
    
    trailing_stop = entry_price - (2 * atr)
    if current_price <= trailing_stop:
        return True, "Trailing Stop Loss"
    
    take_profit = entry_price + (3 * atr)
    if current_price >= take_profit:
        return True, "Take Profit"
    
    days_held = (data.index[-1] - data.index[0]).days
    if days_held >= 5:
        return True, "Time-based Exit"
    
    return False, ""

def get_sector_performance(tickers):
    sector_performance = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            sector = stock.info['sector']
            performance = (stock.history(period="1mo")['Close'].iloc[-1] / stock.history(period="1mo")['Close'].iloc[0] - 1) * 100
            if sector in sector_performance:
                sector_performance[sector].append(performance)
            else:
                sector_performance[sector] = [performance]
        except Exception as e:
            logger.error(f"Error getting sector information for {ticker}: {str(e)}")
    
    # Calculate average performance for each sector
    for sector in sector_performance:
        sector_performance[sector] = sum(sector_performance[sector]) / len(sector_performance[sector])
    
    return sector_performance

def get_news_sentiment(ticker):
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        news = response.json()['articles']
        sentiments = [TextBlob(article['title']).sentiment.polarity for article in news[:5]]
        return sum(sentiments) / len(sentiments) if sentiments else 0
    else:
        logger.error(f"Error fetching news for {ticker}")
        return 0

async def main():
    trades: Dict[str, Dict] = {}

    async with aiohttp.ClientSession() as session:
        logger.info("Fetching combined stock list...")
        tickers = await get_combined_stock_list(session)
        logger.info(f"Combined Stock List: {', '.join(tickers)}")

        if BACKTESTING:
            logger.info("Starting backtesting...")
            for ticker in tickers:
                data = await get_stock_data(session, ticker, TRADING_INTERVAL, TRADING_PERIOD)
                if data.empty:
                    logger.warning(f"No data available for {ticker}")
                    continue

                logger.info(f"Backtesting {ticker} with {len(data)} data points")
                data = calculate_indicators(data)
                if data.empty:
                    logger.warning(f"Insufficient data for calculating indicators for {ticker}")
                if data.empty:
                    logger.warning(f"Insufficient data for calculating indicators for {ticker}")
                    continue

                model = train_ml_model(data)
                
                # Perform backtesting
                balance = ACCOUNT_BALANCE
                position = None
                trades = []

                for i in range(MINIMUM_DATA_POINTS, len(data)):
                    current_data = data.iloc[:i+1]
                    buy, sell, position_size = check_signals(current_data, model, ticker)
                    current_price = current_data['Close'].iloc[-1]

                    if position is None and buy:
                        position = {'entry_price': current_price, 'size': position_size, 'entry_date': current_data.index[-1]}
                        logger.info(f"BUY signal for {ticker} at {position['entry_date']} - Price: {current_price:.2f}, Size: {position_size}")
                    elif position is not None:
                        exit_condition, exit_reason = check_exit_conditions(current_data, position['entry_price'], position['size'])
                        if sell or exit_condition:
                            profit_loss, profit_loss_percentage = calculate_trade_performance(position['entry_price'], current_price)
                            total_pl = profit_loss * position['size']
                            balance += total_pl
                            trades.append({
                                'entry_date': position['entry_date'],
                                'exit_date': current_data.index[-1],
                                'entry_price': position['entry_price'],
                                'exit_price': current_price,
                                'profit_loss': total_pl,
                                'profit_loss_percentage': profit_loss_percentage
                            })
                            logger.info(f"SELL signal for {ticker} at {current_data.index[-1]} - Price: {current_price:.2f}, P/L: ${total_pl:.2f} ({profit_loss_percentage:.2f}%)")
                            position = None

                # Calculate backtesting results
                total_trades = len(trades)
                winning_trades = sum(1 for trade in trades if trade['profit_loss'] > 0)
                total_profit = sum(trade['profit_loss'] for trade in trades)
                max_drawdown = min(trade['profit_loss'] for trade in trades) if trades else 0

                logger.info(f"Backtesting results for {ticker}:")
                logger.info(f"Final Balance: ${balance:.2f}")
                logger.info(f"Total Profit: ${total_profit:.2f}")
                logger.info(f"Total Trades: {total_trades}")
                logger.info(f"Win Rate: {(winning_trades / total_trades * 100 if total_trades > 0 else 0):.2f}%")
                logger.info(f"Max Drawdown: ${abs(max_drawdown):.2f}")

            logger.info("Backtesting completed.")
            return

        # Live trading loop
        while True:
            sector_performance = get_sector_performance(tickers)
            logger.info("Sector Performance:")
            for sector, performance in sector_performance.items():
                logger.info(f"{sector}: {performance:.2f}%")

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

                model = train_ml_model(data)
                buy, sell, position_size = check_signals(data, model, ticker)
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
                        ''', (ticker, trades[ticker]['buy_time'], buy_price, datetime.now(), current_price, total_pl, profit_loss_percentage, int(PAPER_TRADING)))
                        conn.commit()

                        del trades[ticker]

            logger.info(f"Sleeping for {SLEEP_MINUTES} minutes before next update...")
            await asyncio.sleep(SLEEP_MINUTES * 60)

if __name__ == "__main__":
    asyncio.run(main())
