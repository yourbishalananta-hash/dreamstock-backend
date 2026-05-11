from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="MarketPulse Pro API")

# Use a ThreadPoolExecutor to handle synchronous yfinance calls 
# without blocking the FastAPI event loop
executor = ThreadPoolExecutor(max_workers=10)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Technical Indicators (Keep your existing functions) ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "INFY.NS", "SBIN.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "ADANIENT.NS",
    "KOTAKBANK.NS", "TITAN.NS", "ULTRACEMCO.NS", "AXISBANK.NS", "NTPC.NS",
    "ADANIPORTS.NS", "ASIANPAINT.NS", "COALINDIA.NS", "BAJAJFINSV.NS",
    "BPCL.NS", "ONGC.NS", "M&M.NS", "TATAMOTORS.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "ADANIPOWER.NS", "HINDALCO.NS", "GRASIM.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "TECHM.NS", "WIPRO.NS", "INDUSINDBK.NS", "SBILIFE.NS"
]

# NEW: Lightweight function for WebSockets (No heavy indicators)
def get_market_summary():
    """Fetch only latest prices for speed"""
    data = yf.download(TICKERS, period="2d", interval="1d", group_by='ticker', progress=False)
    results = []
    for ticker in TICKERS:
        try:
            # Safe access to multi-index dataframe
            ticker_df = data[ticker].dropna()
            if ticker_df.empty: continue

            close_prices = ticker_df['Close']
            current_price = float(close_prices.iloc[-1])
            prev_price = float(close_prices.iloc[-2])
            change = ((current_price - prev_price) / prev_price) * 100

            results.append({
                "symbol": ticker.replace('.NS', ''),
                "name": ticker.replace('.NS', ''),
                "price": round(current_price, 2),
                "change": round(change, 2),
                "volume": int(ticker_df['Volume'].iloc[-1]),
            })
        except Exception:
            continue
    return {"data": results, "timestamp": datetime.now().isoformat()}

@app.get("/stocks/market-watch")
def get_market_data():
    """Full data with indicators (Keep as is, but wrap in executor if called via async)"""
    df = yf.download(TICKERS, period="2mo", interval="1d", group_by='ticker', progress=False)
    results = []
    for ticker in TICKERS:
        try:
            # Check if ticker exists in result
            if ticker not in df.columns.levels[0]: continue
            
            ticker_df = df[ticker].dropna()
            if ticker_df.empty: continue

            close_prices = ticker_df['Close']
            rsi_values = calculate_rsi(close_prices)
            current_price = float(close_prices.iloc[-1])
            prev_price = float(close_prices.iloc[-2])
            last_rsi = float(rsi_values.iloc[-1])
            change = ((current_price - prev_price) / prev_price) * 100

            results.append({
                "symbol": ticker.replace('.NS', ''),
                "name": ticker.replace('.NS', ''),
                "price": round(current_price, 2),
                "change": round(change, 2),
                "rsi": round(last_rsi, 2) if not np.isnan(last_rsi) else 50,
                "status": "Overbought" if last_rsi > 70 else "Oversold" if last_rsi < 30 else "Neutral",
                "volume": int(ticker_df['Volume'].iloc[-1]),
                "high": float(ticker_df['High'].iloc[-1]),
                "low": float(ticker_df['Low'].iloc[-1]),
            })
        except Exception as e:
            print(f"Error with {ticker}: {e}")
    return {"data": results, "timestamp": datetime.now().isoformat()}

@app.get("/stocks/{symbol}/details")
def get_stock_details(symbol: str):
    ticker = f"{symbol}.NS"
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "symbol": symbol,
        "name": info.get('longName', symbol),
        "sector": info.get('sector', 'N/A'),
        "industry": info.get('industry', 'N/A'),
        "marketCap": info.get('marketCap', 0),
        "peRatio": info.get('trailingPE', 0),
        "eps": info.get('trailingEps', 0),
        "bookValue": info.get('bookValue', 0),
        "dividendYield": info.get('dividendYield', 0),
        "fiftyTwoWeekHigh": info.get('fiftyTwoWeekHigh', 0),
        "fiftyTwoWeekLow": info.get('fiftyTwoWeekLow', 0),
        "beta": info.get('beta', 0),
    }

@app.get("/stocks/{symbol}/technicals")
def get_technicals(symbol: str):
    ticker = f"{symbol}.NS"
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
    if df.empty: raise HTTPException(status_code=404, detail="Stock not found")
    close = df['Close']
    rsi = calculate_rsi(close)
    macd, signal, histogram = calculate_macd(close)
    upper, middle, lower = calculate_bollinger_bands(close)
    return {
        "rsi": round(float(rsi.iloc[-1]), 2),
        "macd": round(float(macd.iloc[-1]), 2),
        "macdSignal": round(float(signal.iloc[-1]), 2),
        "macdHistogram": round(float(histogram.iloc[-1]), 2),
        "bollingerUpper": round(float(upper.iloc[-1]), 2),
        "bollingerMiddle": round(float(middle.iloc[-1]), 2),
        "bollingerLower": round(float(lower.iloc[-1]), 2),
        "sma20": round(float(close.rolling(20).mean().iloc[-1]), 2),
        "sma50": round(float(close.rolling(50).mean().iloc[-1]), 2),
        "sma200": round(float(close.rolling(200).mean().iloc[-1]), 2),
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    loop = asyncio.get_event_loop()
    try:
        while True:
            # FIX: Run synchronous yfinance in a thread to avoid freezing the server
            data = await loop.run_in_executor(executor, get_market_summary)
            await websocket.send_json(data)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/")
def root():
    return {"message": "MarketPulse Pro API is running"}
