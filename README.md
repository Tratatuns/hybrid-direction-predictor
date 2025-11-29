# 🔵 Hybrid Direction Predictor (HDP)

**HDP** is a lightweight, AI-enhanced crypto direction-prediction engine designed to deliver fast, reliable 15-minute LONG/SHORT forecasts and reduce market uncertainty.

It combines:
- advanced technical-indicator logic  
- real-time order-book and exchange data  
- lightweight ML signal generation  

HDP is built for traders, developers, and automated trading systems that need **instant, clean directional signals**.

---

## 🚀 Features

- ⚡ **Ultra-fast 15m LONG/SHORT trend predictions**  
- 📈 Modular technical-indicator engine (EMA, RSI, MACD, ATR, volatility logic)  
- 🔌 Plug-and-play architecture (`run.py`)  
- 🤖 Bot-ready output (JSON, Webhook, API streaming)  
- 🧩 Expandable ML module (LSTM/Transformer upgrade planned)  
- 💧 Upcoming order-flow + liquidity-map analytics  

---

## 🧠 System Architecture

```
/hybrid-direction-predictor
│── core/
│   ├── indicators.py
│   ├── data_pipeline.py
│   ├── predictor.py
│── models/
│   ├── lstm/
│   └── transformer/
│── utils/
│── run.py
│── config.json
│── README.md
```

Modular structure — anything can be replaced or extended.

---

## 🛠 Installation

```
git clone https://github.com/Tratatuns/hybrid-direction-predictor
cd hybrid-direction-predictor
pip install -r requirements.txt
python run.py
```

---

## 🔌 Bot Integration (MEXC / Bybit / Binance / Telegram)

The system outputs **clean LONG/SHORT signals**, which can be connected to:

- Python trading bots  
- TradingView alerts  
- Telegram/Discord signal groups  
- Exchange API auto-execution bots  

---

## 📡 Example Output

```json
{
  "symbol": "BTCUSDT",
  "timeframe": "15m",
  "signal": "LONG",
  "confidence": 0.82,
  "timestamp": "2025-11-29T19:44:12"
}
```

---

## 🎯 Roadmap

- ✅ Technical-indicator engine  
- ✅ Real-time data pipeline  
- 🚧 Order-flow analysis (in progress)  
- 🚧 Liquidity-map visualization  
- ⏳ LSTM/Transformer ML prediction models  
- ⏳ API live-signal broadcasting  

---

## 📺 Demo Video

👉 https://youtu.be/dQw4w9W9gXcQ

---

## 🧑‍💻 Community

👉 Telegram: https://t.me/BalticTradersCrypto

---

## 🔥 Author

**Juris Scerbaks (Hakeris7773)**  
Algoritmiskā tirdzniecība • AI signāli • Crypto market structure  
