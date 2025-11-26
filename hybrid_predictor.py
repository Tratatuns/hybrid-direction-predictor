"""
Hybrid Direction Predictor (HDP)
Simple AI model that predicts the next 15m candle direction (LONG or SHORT)
on BTC/USDT data from Binance.

Usage:
    python hybrid_predictor.py
"""

import ccxt
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import pickle


# ------------------------------------
# 1. Fetch OHLCV data from Binance
# ------------------------------------
def fetch_ohlcv(symbol="BTC/USDT", timeframe="1515m", limit=1500):
    print("Downloading data from Binance...")
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df


# ------------------------------------
# 2. Feature engineering (indicators)
# ------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Calculating technical features...")
    df = df.copy()

    # Returns
    df["ret_1"] = df["close"].pct_change()
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)

    # Volatility (rolling std)
    df["vol_6"] = df["ret_1"].rolling(6).std()
    df["vol_12"] = df["ret_1"].rolling(12).std()

    # EMA indicators
    ema_fast = EMAIndicator(df["close"], window=9).ema_indicator()
    ema_slow = EMAIndicator(df["close"], window=21).ema_indicator()
    df["ema_fast"] = ema_fast
    df["ema_slow"] = ema_slow
    df["ema_diff"] = (ema_fast - ema_slow) / df["close"]

    # RSI indicator
    rsi = RSIIndicator(df["close"], window=14).rsi()
    df["rsi"] = rsi

    # Regime / trend strength
    df["atr_like"] = (df["high"] - df["low"]).rolling(14).mean() / df["close"]
    df["trend_strength"] = df["ema_diff"].abs() / (df["atr_like"] + 1e-8)

    # Target: next candle direction (1 = LONG, 0 = SHORT)
    df["future_close"] = df["close"].shift(-1)
    df["target"] = np.where(df["future_close"] > df["close"], 1, 0)

    df = df.dropna()
    return df


# ------------------------------------
# 3. Train the model
# ------------------------------------
def train_model(df: pd.DataFrame):
    print("Training the AI model...")

    feature_cols = [
        "ret_1", "ret_3", "ret_6",
        "vol_6", "vol_12",
        "ema_diff", "rsi",
        "atr_like", "trend_strength",
    ]

    X = df[feature_cols]
    y = df["target"]

    # 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Test accuracy:", round(acc, 4))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    return model, feature_cols


# ------------------------------------
# 4. Save model to disk
# ------------------------------------
def save_model(model, feature_cols, path="models/hd_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump({"model": model, "features": feature_cols}, f)

    print(f"Model saved to: {path}")


# ------------------------------------
# 5. Predict last candle signal
# ------------------------------------
def predict_last_signal(model, feature_cols, df: pd.DataFrame):
    last_row = df.iloc[-1:][feature_cols]

    prob_long = model.predict_proba(last_row)[0][1]
    signal = "LONG" if prob_long >= 0.5 else "SHORT"

    print(f"\nLast candle signal: {signal} (LONG probability: {prob_long:.3f})")


# ------------------------------------
# 6. Main flow
# ------------------------------------
def main():
    # Step 1: data
    df_raw = fetch_ohlcv()
    os.makedirs("data", exist_ok=True)
    df_raw.to_csv("data/btcusdt_15m_raw.csv")
    print("Raw data saved to data/btcusdt_15m_raw.csv")

    # Step 2: features
    df = add_features(df_raw)
    df.to_csv("data/btcusdt_15m_features.csv")
    print("Feature data saved to data/btcusdt_15m_features.csv")

    # Step 3: train
    model, feature_cols = train_model(df)

    # Step 4: save model
    save_model(model, feature_cols)

    # Step 5: last candle prediction
    predict_last_signal(model, feature_cols, df)


if __name__ == "__main__":
    main()
