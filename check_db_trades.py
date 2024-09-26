import sqlite3
from datetime import datetime

def view_trade_alerts():
    # Connect to the SQLite database
    conn = sqlite3.connect('trading_data.db')
    cursor = conn.cursor()

    # Query to select all trades from the database
    cursor.execute('''
        SELECT * FROM trades
        ORDER BY buy_time DESC
    ''')

    # Fetch all the trades
    trades = cursor.fetchall()

    if not trades:
        print("No trades found in the database.")
    else:
        print(f"Found {len(trades)} trades in the database:")
        print("\n{:<5} {:<10} {:<20} {:<10} {:<20} {:<10} {:<10} {:<10} {:<15}".format(
            "ID", "Ticker", "Buy Time", "Buy Price", "Sell Time", "Sell Price", "P/L", "P/L %", "Paper Trade"
        ))
        print("-" * 120)

        for trade in trades:
            trade_id, ticker, buy_time, buy_price, sell_time, sell_price, profit_loss, profit_loss_percentage, is_paper_trade = trade
            
            # Convert timestamps to readable format
            buy_time = datetime.fromisoformat(buy_time).strftime('%Y-%m-%d %H:%M:%S')
            sell_time = datetime.fromisoformat(sell_time).strftime('%Y-%m-%d %H:%M:%S') if sell_time else "N/A"

            print("{:<5} {:<10} {:<20} {:<10.2f} {:<20} {:<10.2f} {:<10.2f} {:<10.2f} {:<15}".format(
                trade_id, ticker, buy_time, buy_price, sell_time, sell_price or 0,
                profit_loss or 0, profit_loss_percentage or 0,
                "Yes" if is_paper_trade else "No"
            ))

    # Close the database connection
    conn.close()

if __name__ == "__main__":
    view_trade_alerts()
