import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import talib
import matplotlib.pyplot as plt
import seaborn as sns
import traceback

warnings.filterwarnings("ignore")

FUTURES_SPECS = {
    "GC=F": {"name": "Gold", "contract_size": 100, "tick_size": 0.10, "margin": 11000},
    "SI=F": {"name": "Silver", "contract_size": 5000, "tick_size": 0.005, "margin": 14000},
    "CL=F": {"name": "Crude Oil", "contract_size": 1000, "tick_size": 0.01, "margin": 6500},
    "NG=F": {"name": "Natural Gas", "contract_size": 10000, "tick_size": 0.001, "margin": 2200},
    "ZC=F": {"name": "Corn", "contract_size": 5000, "tick_size": 0.25, "margin": 2200},
    "ZS=F": {"name": "Soybeans", "contract_size": 5000, "tick_size": 0.25, "margin": 3400},
    "KC=F": {"name": "Coffee", "contract_size": 37500, "tick_size": 0.05, "margin": 3300},
    "CT=F": {"name": "Cotton", "contract_size": 50000, "tick_size": 0.01, "margin": 2500}
}

def fetch_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            print(f"No data available for {symbol}")
            return None
        return data
    except Exception as e:
        print(f"Error fetching data for {symbol}: {str(e)}")
        return None

def feature_engineering(df):
    print("Feature engineering: Initial shape:", df.shape)
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = df['Returns'].rolling(window=20).std()
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_Signal'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['ATR'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['CCI'] = talib.CCI(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['ROC'] = talib.ROC(df['Close'], timeperiod=10)
    df['MOM'] = talib.MOM(df['Close'], timeperiod=10)

    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_30'] = df['Close'].rolling(window=30).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_30'] = df['Close'].ewm(span=30, adjust=False).mean()

    df['Upper_BB'], df['Middle_BB'], df['Lower_BB'] = talib.BBANDS(df['Close'], timeperiod=20)
    df['BB_Width'] = (df['Upper_BB'] - df['Lower_BB']) / df['Middle_BB']

    df['Price_Direction_1d'] = df['Close'].shift(-1) > df['Close']
    df['Price_Direction_5d'] = df['Close'].shift(-5) > df['Close']
    df['Price_Direction_20d'] = df['Close'].shift(-20) > df['Close']

    df['Day_of_Week'] = df.index.dayofweek
    df['Month'] = df.index.month

    print("Before dropna:", df.shape)
    df = df.dropna()
    print("After dropna:", df.shape)

    df['Price_Direction_1d'] = df['Price_Direction_1d'].astype(int)
    df['Price_Direction_5d'] = df['Price_Direction_5d'].astype(int)
    df['Price_Direction_20d'] = df['Price_Direction_20d'].astype(int)

    print("Feature engineering: Final shape:", df.shape)
    return df

def prepare_data(df):
    features = ['Returns', 'Log_Returns', 'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'ATR',
                'ADX', 'CCI', 'ROC', 'MOM', 'SMA_10', 'SMA_30', 'EMA_10', 'EMA_30',
                'Upper_BB', 'Middle_BB', 'Lower_BB', 'BB_Width', 'Day_of_Week', 'Month']

    print("Columns in df:", df.columns)
    print("Shape of df:", df.shape)

    X = df[features + ['Close']]
    y_1d = df['Price_Direction_1d']
    y_5d = df['Price_Direction_5d']
    y_20d = df['Price_Direction_20d']

    print("Shape of X:", X.shape)
    print("Shape of y_1d:", y_1d.shape)
    print("Shape of y_5d:", y_5d.shape)
    print("Shape of y_20d:", y_20d.shape)

    return X, y_1d, y_5d, y_20d

def train_model(X, y):
    X_model = X.drop('Close', axis=1)
    tscv = TimeSeriesSplit(n_splits=5)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X_model)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    scores = []

    for train_index, test_index in tscv.split(X_pca):
        X_train, X_test = X_pca[train_index], X_pca[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        scores.append({
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        })

    avg_scores = {k: np.mean([d[k] for d in scores]) for k in scores[0]}

    model.fit(X_pca, y)

    return model, imputer, scaler, pca, avg_scores

def get_feature_importance(model, pca, feature_names):
    importances = model.feature_importances_

    mask = [col != 'Close' for col in feature_names]

    original_importances = np.zeros(len(feature_names))

    for i, importance in enumerate(importances):
        original_importances[mask] += importance * pca.components_[i]

    original_importances = original_importances / np.sum(original_importances)

    forest_importances = pd.Series(original_importances, index=feature_names)
    return forest_importances.sort_values(ascending=False)

def plot_feature_importance(importances, title):
    plt.figure(figsize=(10, 6))
    importances.plot.bar()
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()

def calculate_position_size(capital, risk_per_trade, confidence, volatility, margin):
    risk_amount = capital * risk_per_trade / 100
    position_size = risk_amount * confidence / (volatility * 100)
    num_contracts = int(position_size / margin)
    return max(1, min(num_contracts, 5))

def get_market_sentiment(symbol, start_date, end_date):
    spy_data = yf.download("SPY", start=start_date, end=end_date)
    if spy_data.empty:
        return 0
    spy_returns = spy_data['Close'].pct_change().dropna()
    sentiment = np.sign(spy_returns.mean())
    return sentiment

def backtest(symbol, model, imputer, scaler, pca, X, y, capital, risk_per_trade):
    X_model = X.drop('Close', axis=1)
    X_imputed = imputer.transform(X_model)
    X_scaled = scaler.transform(X_imputed)
    X_pca = pca.transform(X_scaled)

    predictions = model.predict(X_pca)
    probabilities = model.predict_proba(X_pca)[:, 1]

    portfolio_value = [capital]
    positions = []

    for i in range(len(X)):
        if i == 0:
            positions.append(0)
            continue

        confidence = abs(probabilities[i] - 0.5) * 2
        volatility = X['Volatility'].iloc[i]
        margin = FUTURES_SPECS.get(symbol, {"margin": 10000})["margin"]

        if confidence > 0.6:
            if probabilities[i] > 0.5:
                num_contracts = calculate_position_size(portfolio_value[-1], risk_per_trade, confidence, volatility, margin)
                positions.append(num_contracts)
            else:
                num_contracts = calculate_position_size(portfolio_value[-1], risk_per_trade, confidence, volatility, margin)
                positions.append(-num_contracts)
        else:
            positions.append(0)

        contract_size = FUTURES_SPECS.get(symbol, {"contract_size": 100})["contract_size"]
        pnl = positions[i-1] * (X['Close'].iloc[i] - X['Close'].iloc[i-1]) * contract_size
        portfolio_value.append(portfolio_value[-1] + pnl)

    return portfolio_value, positions

def plot_backtest_results(X, portfolio_value, positions):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    ax1.plot(X.index, X['Close'])
    ax1.set_title('Price')
    ax1.set_ylabel('Price')

    ax2.plot(X.index, portfolio_value)
    ax2.set_title('Portfolio Value')
    ax2.set_ylabel('Value')

    ax3.plot(X.index, positions)
    ax3.set_title('Position Size')
    ax3.set_ylabel('Number of Contracts')

    plt.tight_layout()
    plt.show()

def analyze_commodity(symbol, data, capital, risk_per_trade):
    try:
        print(f"\nAnalyzing {symbol}")
        print(f"Initial data shape: {data.shape}")

        data = feature_engineering(data)
        print(f"After feature engineering shape: {data.shape}")

        X, y_1d, y_5d, y_20d = prepare_data(data)
        print(f"X shape: {X.shape}")
        print(f"y_1d shape: {y_1d.shape}")
        print(f"y_5d shape: {y_5d.shape}")
        print(f"y_20d shape: {y_20d.shape}")

        models = {}
        imputers = {}
        scalers = {}
        pcas = {}
        scores = {}
        importances = {}

        for horizon, y in [('1d', y_1d), ('5d', y_5d), ('20d', y_20d)]:
            print(f"\nTraining model for {horizon} horizon")
            model, imputer, scaler, pca, avg_scores = train_model(X, y)
            models[horizon] = model
            imputers[horizon] = imputer
            scalers[horizon] = scaler
            pcas[horizon] = pca
            scores[horizon] = avg_scores
            importances[horizon] = get_feature_importance(model, pca, X.columns)

        latest_features = X.iloc[-1:]
        predictions = {}
        confidences = {}

        for horizon in ['1d', '5d', '20d']:
            latest_model_features = latest_features.drop('Close', axis=1)
            latest_imputed = imputers[horizon].transform(latest_model_features)
            latest_scaled = scalers[horizon].transform(latest_imputed)
            latest_pca = pcas[horizon].transform(latest_scaled)
            prob = models[horizon].predict_proba(latest_pca)[0][1]
            predictions[horizon] = "BUY" if prob > 0.5 else "SELL"
            confidences[horizon] = min(abs(prob - 0.5) * 2, 0.95)

        current_price = data['Close'].iloc[-1]
        contract_spec = FUTURES_SPECS.get(symbol, {"name": symbol})

        print(f"\nAnalysis for {contract_spec['name']} ({symbol}):")
        print(f"Current Price: ${current_price:.2f}")

        for horizon in ['1d', '5d', '20d']:
            print(f"\n{horizon} Prediction:")
            print(f"Signal: {predictions[horizon]}")
            print(f"Confidence: {confidences[horizon]:.2f}")
            print(f"Model Performance:")
            print(f"Accuracy: {scores[horizon]['accuracy']:.4f}, F1: {scores[horizon]['f1']:.4f}")

        for horizon in ['1d', '5d', '20d']:
            plot_feature_importance(importances[horizon], f"Feature Importance - {horizon} Horizon")

        portfolio_value, positions = backtest(symbol, models['1d'], imputers['1d'], scalers['1d'], pcas['1d'], X, y_1d, capital, risk_per_trade)
        plot_backtest_results(data, portfolio_value, positions)

        sentiment = get_market_sentiment(symbol, data.index[0], data.index[-1])
        print(f"\nMarket Sentiment: {'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'}")

        return predictions, confidences, scores
    except Exception as e:
        print(f"Analysis failed for {symbol}: {str(e)}")
        traceback.print_exc()
        return {"1d": "HOLD", "5d": "HOLD", "20d": "HOLD"}, {"1d": 0, "5d": 0, "20d": 0}, {}

def analyze_multiple_commodities(commodities, capital, risk_per_trade):
    start_date = (datetime.now() - timedelta(days=1000)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    all_data = {}
    all_predictions = {}
    all_confidences = {}
    all_scores = {}

    for symbol in commodities:
        data = fetch_data(symbol, start_date, end_date)
        if data is not None and not data.empty:
            all_data[symbol] = data['Close']
            predictions, confidences, scores = analyze_commodity(symbol, data, capital, risk_per_trade)
            all_predictions[symbol] = predictions
            all_confidences[symbol] = confidences
            all_scores[symbol] = scores

            # Provide summary for each commodity
            provide_beginner_summary(symbol, predictions, confidences, data['Close'].iloc[-1], scores)

    if all_data:
        correlation_matrix = pd.DataFrame(all_data).corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of Commodities')
        plt.show()

        reported_pairs = set()
        for symbol1, preds1 in all_predictions.items():
            for symbol2, preds2 in all_predictions.items():
                if symbol1 != symbol2 and (symbol2, symbol1) not in reported_pairs:
                    correlation = correlation_matrix.loc[symbol1, symbol2]
                    if abs(correlation) > 0.7:
                        print(f"\nWarning: High correlation ({correlation:.2f}) between {symbol1} and {symbol2}")
                        for horizon in ['1d', '5d', '20d']:
                            if preds1[horizon] == preds2[horizon] and preds1[horizon] != "HOLD":
                                print(f"Both have {preds1[horizon]} signals for {horizon} horizon.")
                        print("Consider diversifying.")
                        reported_pairs.add((symbol1, symbol2))

    print("\nIMPORTANT WARNINGS:")
    print("1. This model predicts price movement direction, not exact prices.")
    print("2. Past performance does not guarantee future results.")
    print("3. Transaction costs can significantly impact profitability.")
    print("4. This script is for educational purposes only and should not be used for real trading without significant further development and risk management implementation.")
    print("5. Always consult with financial professionals before making investment decisions.")
    print("6. Ensure compliance with all relevant financial regulations in your jurisdiction.")

def provide_beginner_summary(symbol, predictions, confidences, current_price, scores):
    print("\n=== Beginner-Friendly Summary ===")
    print(f"Analysis for {symbol}:")
    print(f"Current Price: ${current_price:.2f}")

    for horizon in ['1d', '5d', '20d']:
        print(f"\n{horizon} Prediction:")
        print(f"The model suggests to {predictions[horizon]} with {confidences[horizon]:.2f} confidence.")
        print(f"Model Accuracy: {scores[horizon]['accuracy']:.2f}")

    print("\nWhat does this mean?")
    print("- A 'BUY' suggestion means the model thinks the price might go up.")
    print("- A 'SELL' suggestion means the model thinks the price might go down.")
    print("- Confidence ranges from 0 to 1. Higher numbers mean the model is more sure.")
    print("- Model Accuracy shows how often the model has been right in tests.")

    print("\nSuggestions:")
    print("1. Don't make decisions based solely on this model. It's just one tool.")
    print("2. Higher confidence and accuracy are generally better, but not guaranteed.")
    print("3. Consider looking at longer time horizons (like 20d) for more stable predictions.")
    print("4. Always do your own research and consider multiple sources of information.")
    print("5. Start with small investments and learn as you go.")
    print("6. Remember that all investments carry risk. Never invest more than you can afford to lose.")

def analyze_user_symbol(symbol, capital=100000, risk_per_trade=1):
    try:
        start_date = (datetime.now() - timedelta(days=1000)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        data = fetch_data(symbol, start_date, end_date)
        if data is None or data.empty:
            print(f"No data available for {symbol}. Please check the ticker symbol and try again.")
            return

        predictions, confidences, scores = analyze_commodity(symbol, data, capital, risk_per_trade)

        current_price = data['Close'].iloc[-1]

        provide_beginner_summary(symbol, predictions, confidences, current_price, scores)

    except Exception as e:
        print(f"An error occurred while analyzing {symbol}: {str(e)}")
        traceback.print_exc()

def main():
    while True:
        print("\n=== Commodity Analysis Tool ===")
        print("1. Analyze a specific symbol")
        print("2. Analyze all predefined commodities")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ")

        if choice == '1':
            symbol = input("Enter the ticker symbol to analyze (e.g., AAPL, GOOGL): ").upper()
            analyze_user_symbol(symbol)
        elif choice == '2':
            capital = 100000
            risk_per_trade = 1
            analyze_multiple_commodities(FUTURES_SPECS.keys(), capital, risk_per_trade)
        elif choice == '3':
            print("Thank you for using the Commodity Analysis Tool. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
