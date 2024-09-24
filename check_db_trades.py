import sqlite3
from datetime import datetime, timedelta

def connect_to_db():
    return sqlite3.connect('trading_data.db')

def fetch_trades(conn, days_ago=None):
    cursor = conn.cursor()
    
    if days_ago is not None:
        start_date = datetime.now() - timedelta(days=days_ago)
        cursor.execute('''
            SELECT * FROM trades
            WHERE buy_time > ?
            ORDER BY buy_time DESC
        ''', (start_date,))
    else:
        cursor.execute('SELECT * FROM trades ORDER BY buy_time DESC')
    
    return cursor.fetchall()

def display_trades(trades):
    if not trades:
        print("No trades found.")
        return

    print(f"Found {len(trades)} trade(s):")
    print("-" * 80)
    for trade in trades:
        print(f"Ticker: {trade[1]}")
        print(f"Buy Time: {trade[2]}")
        print(f"Buy Price: ${trade[3]:.2f}")
        print(f"Sell Time: {trade[4]}")
        print(f"Sell Price: ${trade[5]:.2f}")
        print(f"Profit/Loss: ${trade[6]:.2f}")
        print(f"Profit/Loss %: {trade[7]:.2f}%")
        print(f"Paper Trade: {'Yes' if trade[8] else 'No'}")
        print("-" * 80)

def main():
    conn = connect_to_db()
    
    while True:
        print("\nOptions:")
        print("1. View all trades")
        print("2. View trades from the last X days")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            trades = fetch_trades(conn)
            display_trades(trades)
        elif choice == '2':
            days = int(input("Enter the number of days to look back: "))
            trades = fetch_trades(conn, days)
            display_trades(trades)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please try again.")
    
    conn.close()
    print("Database connection closed. Goodbye!")

if __name__ == "__main__":
    main()
