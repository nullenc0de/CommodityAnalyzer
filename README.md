=== Commodity Analysis Tool ===
1. Analyze a specific symbol
2. Analyze all predefined commodities
3. Exit
Enter your choice (1-3): 1
Enter the ticker symbol to analyze (e.g., AAPL, GOOGL): f
[*********************100%***********************]  1 of 1 completed

Analyzing F
Initial data shape: (686, 6)
Feature engineering: Initial shape: (686, 6)
Before dropna: (686, 30)
After dropna: (653, 30)
Feature engineering: Final shape: (653, 30)
After feature engineering shape: (653, 30)
Columns in df: Index(['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Returns',
       'Log_Returns', 'Volatility', 'RSI', 'MACD', 'MACD_Signal', 'ATR', 'ADX',
       'CCI', 'ROC', 'MOM', 'SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'Upper_BB',
       'Middle_BB', 'Lower_BB', 'BB_Width', 'Price_Direction_1d',
       'Price_Direction_5d', 'Price_Direction_20d', 'Day_of_Week', 'Month'],
      dtype='object')
Shape of df: (653, 30)
Shape of X: (653, 22)
Shape of y_1d: (653,)
Shape of y_5d: (653,)
Shape of y_20d: (653,)
X shape: (653, 22)
y_1d shape: (653,)
y_5d shape: (653,)
y_20d shape: (653,)

Training model for 1d horizon

Training model for 5d horizon

Training model for 20d horizon

Analysis for F (F):
Current Price: $10.80

1d Prediction:
Signal: SELL
Confidence: 0.11
Model Performance:
Accuracy: 0.5278, F1: 0.5607

5d Prediction:
Signal: SELL
Confidence: 0.43
Model Performance:
Accuracy: 0.5963, F1: 0.6015

20d Prediction:
Signal: SELL
Confidence: 0.38
Model Performance:
Accuracy: 0.6685, F1: 0.6533
[*********************100%***********************]  1 of 1 completed

Market Sentiment: Positive

=== Beginner-Friendly Summary ===
Analysis for F:
Current Price: $10.80

1d Prediction:
The model suggests to SELL with 0.11 confidence.
Model Accuracy: 0.53

5d Prediction:
The model suggests to SELL with 0.43 confidence.
Model Accuracy: 0.60

20d Prediction:
The model suggests to SELL with 0.38 confidence.
Model Accuracy: 0.67

What does this mean?
- A 'BUY' suggestion means the model thinks the price might go up.
- A 'SELL' suggestion means the model thinks the price might go down.
- Confidence ranges from 0 to 1. Higher numbers mean the model is more sure.
- Model Accuracy shows how often the model has been right in tests.

Suggestions:
1. Don't make decisions based solely on this model. It's just one tool.
2. Higher confidence and accuracy are generally better, but not guaranteed.
3. Consider looking at longer time horizons (like 20d) for more stable predictions.
4. Always do your own research and consider multiple sources of information.
5. Start with small investments and learn as you go.
6. Remember that all investments carry risk. Never invest more than you can afford to lose.
