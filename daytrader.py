import yfinance as yf
import pandas as pd
import numpy as np
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def get_stock_data(ticker, interval, period):
    stock = yf.download(ticker, period=period, interval=interval)
    return stock

def calculate_indicators(data):
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = get_rsi(data['Close'])
    data['MACD'], data['MACD_Signal'], _ = get_macd(data['Close'])
    return data

def get_rsi(close, timeperiod=14):
    diff = close.diff()
    up = diff.clip(lower=0)
    down = -diff.clip(upper=0)
    ema_up = up.ewm(com=timeperiod-1, adjust=True, min_periods=timeperiod).mean()
    ema_down = down.ewm(com=timeperiod-1, adjust=True, min_periods=timeperiod).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_macd(close, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = close.ewm(span=fast_period, adjust=False).mean()
    ema_slow = close.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal_period, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def check_signals(data):
    if len(data) < 2:
        return False, False
    
    buy_signal = (data['Close'].iloc[-1] > data['SMA_20'].iloc[-1]) and (data['RSI'].iloc[-1] < 30) and (data['MACD'].iloc[-1] > data['MACD_Signal'].iloc[-1])
    sell_signal = (data['Close'].iloc[-1] < data['SMA_50'].iloc[-1]) and (data['RSI'].iloc[-1] > 70) and (data['MACD'].iloc[-1] < data['MACD_Signal'].iloc[-1])
    return buy_signal, sell_signal

def get_most_active_stocks():
    url = 'https://stockanalysis.com/markets/active/'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', {'id': 'main-table'})
    rows = table.find_all('tr')[1:]  # Exclude the header row
    symbols = []
    for row in rows:
        symbol = row.find_all('td')[1].text.strip()
        symbols.append(symbol)
    return symbols

def calculate_trade_performance(ticker, buy_price, sell_price):
    profit_loss = sell_price - buy_price
    profit_loss_percentage = (profit_loss / buy_price) * 100
    return profit_loss, profit_loss_percentage

def main():
    interval = '5m'
    period = '1d'
    check_interval = timedelta(hours=1)
    trades = {}
    
    while True:
        print(f"\n{datetime.now()} - Fetching most active stocks...")
        tickers = get_most_active_stocks()
        print(f"Most Active Stocks: {', '.join(tickers)}")
        
        for ticker in tickers:
            try:
                data = get_stock_data(ticker, interval, period)
                data = calculate_indicators(data)
                buy, sell = check_signals(data)
                current_price = data['Close'].iloc[-1]
                
                if buy and ticker not in trades:
                    trades[ticker] = {'buy_time': datetime.now(), 'buy_price': current_price}
                    print(f"BUY signal for {ticker} at price {current_price}")
                elif sell and ticker in trades:
                    buy_price = trades[ticker]['buy_price']
                    profit_loss, profit_loss_percentage = calculate_trade_performance(ticker, buy_price, current_price)
                    print(f"SELL signal for {ticker} at price {current_price}")
                    print(f"Trade performance for {ticker}: Profit/Loss = {profit_loss:.2f}, Profit/Loss % = {profit_loss_percentage:.2f}%")
                    del trades[ticker]
                    
                # Check if any open trades have reached the 1-hour mark
                for ticker, trade in list(trades.items()):
                    if datetime.now() - trade['buy_time'] >= check_interval:
                        buy_price = trade['buy_price']
                        profit_loss, profit_loss_percentage = calculate_trade_performance(ticker, buy_price, current_price)
                        print(f"Closing trade for {ticker} after 1 hour. Profit/Loss = {profit_loss:.2f}, Profit/Loss % = {profit_loss_percentage:.2f}%")
                        del trades[ticker]
                        
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
        
        print(f"Sleeping for 5 minutes before next update...")
        time.sleep(300)  # Wait for 5 minutes before the next update

if __name__ == "__main__":
    main()
