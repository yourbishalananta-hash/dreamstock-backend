from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# FIXED RSI calculation
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    # Handle division by zero
    rs = gain / loss.replace(0, np.nan) 
    return 100 - (100 / (1 + rs))

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "INFY.NS", "SBIN.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "ADANIENT.NS",
    "KOTAKBANK.NS", "TITAN.NS", "ULTRACEMCO.NS", "AXISBANK.NS", "NTPC.NS",
    "ADANIPORTS.NS", "ASIANPAINT.NS", "COALINDIA.NS", "BAJAJFINSV.NS", 
    "BPCL.NS", "ONGC.NS", "M&M.NS", "TATAMOTORS.NS", "JSWSTEEL.NS",
    "TATASTEEL.NS", "ADANIPOWER.NS", "HINDALCO.NS", "GRASIM.NS", "NESTLEIND.NS",
    "POWERGRID.NS", "TECHM.NS", "WIPRO.NS", "INDUSINDBK.NS", "SBILIFE.NS"]

@app.get("/stocks/market-watch")
def get_market_data():
    # Download data for all tickers at once
    df = yf.download(TICKERS, period="2mo", interval="1d", group_by='ticker')
    
    results = []
    for ticker in TICKERS:
        try:
            # Handle the multi-index dataframe correctly
            ticker_df = df[ticker].dropna()
            if ticker_df.empty: continue

            close_prices = ticker_df['Close']
            rsi_values = calculate_rsi(close_prices)
            
            current_price = float(close_prices.iloc[-1])
            last_rsi = float(rsi_values.iloc[-1])

            results.append({
                "symbol": ticker,
                "price": round(current_price, 2),
                "rsi": round(last_rsi, 2) if not np.isnan(last_rsi) else "N/A",
                "status": "Overbought" if last_rsi > 70 else "Oversold" if last_rsi < 30 else "Neutral"
            })
        except Exception as e:
            print(f"Error with {ticker}: {e}")
            
    return {"data": results}

@app.get("/")
def root():
    # This MUST be indented (tabbed in) to stay inside the function
    return {"message": "API is online and syntax error is fixed!"}