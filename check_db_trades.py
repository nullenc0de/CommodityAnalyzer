import sqlite3

def fetch_all_trades():
    # Connect to the SQLite database
    conn = sqlite3.connect('trading_data.db')
    cursor = conn.cursor()

    try:
        # Execute a query to select all records from the trades table
        cursor.execute("SELECT * FROM trades")
        rows = cursor.fetchall()

        # Print the results
        if rows:
            for row in rows:
                print(row)
        else:
            print("No records found in the trades table.")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
    finally:
        # Close the database connection
        conn.close()

if __name__ == "__main__":
    fetch_all_trades()
