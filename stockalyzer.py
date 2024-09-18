import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
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

    df['OBV'] = talib.OBV(df['Close'], df['Volume'])
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
    df['WILLR'] = talib.WILLR(df['High'], df['Low'], df['Close'])

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
                'Upper_BB', 'Middle_BB', 'Lower_BB', 'BB_Width', 'Day_of_Week', 'Month',
                'OBV', 'STOCH_K', 'STOCH_D', 'WILLR']

    print("Columns in df:", df.columns)
    print("Shape of df:", df.shape)

    X = df[features]
    y_1d = df['Price_Direction_1d']
    y_5d = df['Price_Direction_5d']
    y_20d = df['Price_Direction_20d']

    print("Shape of X:", X.shape)
    print("Shape of y_1d:", y_1d.shape)
    print("Shape of y_5d:", y_5d.shape)
    print("Shape of y_20d:", y_20d.shape)

    return X, y_1d, y_5d, y_20d

def train_ensemble_model(X, y):
    tscv = TimeSeriesSplit(n_splits=5)

    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_scaled)

    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0),
        'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'LogisticRegression': LogisticRegression(random_state=42)
    }

    param_grids = {
        'RandomForest': {'n_estimators': [50, 100], 'max_depth': [3, 5]},
        'GradientBoosting': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]},
        'XGBoost': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]},
        'LightGBM': {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]},
        'LogisticRegression': {'C': [0.001, 0.01, 0.1, 1]}
    }

    best_models = {}
    best_scores = {}

    for name, model in models.items():
        print(f"Training {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=tscv, scoring='f1', n_jobs=-1)
        grid_search.fit(X_pca, y)
        best_models[name] = grid_search.best_estimator_
        best_scores[name] = grid_search.best_score_
        print(f"{name} best F1 score: {best_scores[name]:.4f}")

    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in best_models.items()],
        voting='soft'
    )
    ensemble.fit(X_pca, y)

    return ensemble, imputer, scaler, pca, best_scores

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

        for horizon, y in [('1d', y_1d), ('5d', y_5d), ('20d', y_20d)]:
            print(f"\nTraining model for {horizon} horizon")
            model, imputer, scaler, pca, avg_scores = train_ensemble_model(X, y)
            models[horizon] = model
            imputers[horizon] = imputer
            scalers[horizon] = scaler
            pcas[horizon] = pca
            scores[horizon] = avg_scores

        latest_features = X.iloc[-1:]
        predictions = {}
        confidences = {}

        for horizon in ['1d', '5d', '20d']:
            latest_imputed = imputers[horizon].transform(latest_features)
            latest_scaled = scalers[horizon].transform(latest_imputed)
            latest_pca = pcas[horizon].transform(latest_scaled)
            prob = models[horizon].predict_proba(latest_pca)[0][1]

            # Adjust confidence calculation
            raw_confidence = abs(prob - 0.5) * 2
            adjusted_confidence = 1 / (1 + np.exp(-10 * (raw_confidence - 0.5)))

            if adjusted_confidence > 0.6:
                predictions[horizon] = "BUY" if prob > 0.5 else "SELL"
            else:
                predictions[horizon] = "HOLD"

            confidences[horizon] = adjusted_confidence

        current_price = data['Close'].iloc[-1]
        contract_spec = FUTURES_SPECS.get(symbol, {"name": symbol})

        print(f"\nAnalysis for {contract_spec['name']} ({symbol}):")
        print(f"Current Price: ${current_price:.2f}")

        for horizon in ['1d', '5d', '20d']:
            print(f"\n{horizon} Prediction:")
            print(f"Signal: {predictions[horizon]}")
            print(f"Confidence: {confidences[horizon]:.2f}")
            print(f"Model Performance:")
            for model_name, score in scores[horizon].items():
                print(f"{model_name} F1 Score: {score:.4f}")

        # Use RandomForest for feature importance as VotingClassifier doesn't have feature_importances_
        rf_model = models['1d'].estimators_[0]  # Assuming RandomForest is the first estimator
        plot_feature_importance(rf_model, X.columns, "Feature Importance - 1d Horizon (RandomForest)")

        return predictions, confidences, scores
    except Exception as e:
        print(f"Analysis failed for {symbol}: {str(e)}")
        traceback.print_exc()
        return {"1d": "HOLD", "5d": "HOLD", "20d": "HOLD"}, {"1d": 0, "5d": 0, "20d": 0}, {}

def plot_feature_importance(model, feature_names, title):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10,6))
        plt.title(title)
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("Model doesn't have feature_importances_ attribute. Skipping feature importance plot.")

def provide_beginner_summary(symbol, predictions, confidences, current_price, scores):
    print("\n=== Investment Opportunity Summary ===")
    print(f"Analysis for {symbol}:")
    print(f"Current Price: ${current_price:.2f}")

    for horizon in ['1d', '5d', '20d']:
        print(f"\n{horizon} Outlook:")
        print(f"Model suggestion: {predictions[horizon]}")
        print(f"Confidence: {confidences[horizon]:.2f}")
        if horizon in scores:
            print(f"Average Model F1 Score: {np.mean(list(scores[horizon].values())):.2f}")
        else:
            print("Model performance data not available.")

        if predictions[horizon] == "BUY" and confidences[horizon] > 0.6:
            print(f"Potential opportunity: Consider buying for a {horizon} hold.")
        elif predictions[horizon] == "SELL" and confidences[horizon] > 0.6:
            print(f"Potential opportunity: Consider selling if you own this asset.")
        else:
            print("No clear opportunity identified.")

    print("\nIMPORTANT DISCLAIMER:")
    print("These suggestions are based on a predictive model and should NOT be the sole basis for investment decisions.")
    print("Always conduct your own research and consider seeking professional financial advice before trading.")
    print("Remember that all investments carry risk, including the potential loss of principal.")


    print("\nWhat does this mean?")
    print("- A 'BUY' suggestion means the model thinks the price might go up.")
    print("- A 'SELL' suggestion means the model thinks the price might go down.")
    print("- A 'HOLD' suggestion means the model is not confident enough to suggest a buy or sell.")
    print("- Confidence ranges from 0 to 1. Higher numbers mean the model is more sure.")
    print("- F1 Score is a measure of the model's accuracy. It ranges from 0 to 1, where 1 is perfect.")

    print("\nRisk Management Suggestions:")
    print("1. Don't make decisions based solely on this model. It's just one tool.")
    print("2. Always use stop-loss orders to limit potential losses.")
    print("3. Never invest more than you can afford to lose.")
    print("4. Consider the overall market conditions and news that might affect the asset.")
    print("5. Diversify your investments to spread risk.")
    print("6. Be aware that short-term predictions (1d) are generally less reliable than longer-term ones.")
    print("7. Monitor your positions regularly and be prepared to exit if conditions change.")

    print("\nFurther Steps:")
    print("1. Research the fundamentals of the asset you're interested in.")
    print("2. Look at longer-term trends and overall market conditions.")
    print("3. Consider consulting with a financial advisor for personalized advice.")
    print("4. Keep learning about different investment strategies and risk management techniques.")

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
            for symbol in FUTURES_SPECS.keys():
                analyze_user_symbol(symbol, capital, risk_per_trade)
        elif choice == '3':
            print("Thank you for using the Commodity Analysis Tool. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
