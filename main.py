from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="MarketPulse Pro Enterprise API")

# --- CONFIGURATION ---
# In a real scenario, load this from a CSV of all NSE symbols
# For this demo, we use a larger sample, but the architecture supports 5000+
ALL_INDIAN_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "INFY.NS", "SBIN.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "ADANIENT.NS",
    # ... Imagine 2000 more symbols here
]

# --- CACHE SYSTEM ---
# This prevents the server from hitting Yahoo Finance on every single request
class MarketCache:
    def __init__(self):
        self.snapshot = {}       # LTP, Change, Volume
        self.fundamentals = {}   # PE, MarketCap, BasePrice
        self.last_updated = None

    def update_snapshot(self, data):
        self.snapshot = data
        self.last_updated = datetime.now()

    def get_all_stocks(self, page: int, limit: int):
        symbols = list(self.snapshot.keys())
        start = (page - 1) * limit
        end = start + limit
        paginated_symbols = symbols[start:end]
        
        return [self.snapshot[s] for s in paginated_symbols]

cache = MarketCache()
executor = ThreadPoolExecutor(max_workers=20)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CORE DATA FETCHERS ---

def fetch_all_market_data():
    """
    Optimized batch fetcher. 
    Fetches data for all tickers in one large request to minimize API hits.
    """
    try:
        # Fetching 2 days of data for all symbols to get LTP and Prev Close (Base Price)
        data = yf.download(ALL_INDIAN_TICKERS, period="2d", interval="1d", group_by='ticker', progress=False)
        
        snapshot = {}
        for ticker in ALL_INDIAN_TICKERS:
            try:
                t_df = data[ticker].dropna()
                if t_df.empty: continue
                
                # Financial Definitions:
                # LTP: Last Traded Price (The most recent close)
                # Base Price: The previous closing price
                ltp = float(t_df['Close'].iloc[-1])
                base_price = float(t_df['Close'].iloc[-2]) 
                change = ((ltp - base_price) / base_price) * 100
                volume = int(t_df['Volume'].iloc[-1])

                snapshot[ticker] = {
                    "symbol": ticker.replace('.NS', ''),
                    "ltp": round(ltp, 2),
                    "basePrice": round(base_price, 2),
                    "change": round(change, 2),
                    "volume": volume,
                    "timestamp": datetime.now().isoformat()
                }
            except:
                continue
        return snapshot
    except Exception as e:
        print(f"Batch Fetch Error: {e}")
        return {}

# --- BACKGROUND WORKER ---

async def market_data_worker():
    """
    The Heartbeat of the Server.
    Updates the cache every 60 seconds so users get instant data.
    """
    while True:
        print("Updating Global Market Cache...")
        loop = asyncio.get_event_loop()
        # Run the heavy yfinance call in the thread pool
        new_data = await loop.run_in_executor(executor, fetch_all_market_data)
        if new_data:
            cache.update_snapshot(new_data)
        
        # Sleep for 60 seconds to avoid Rate Limiting
        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    # Start the background worker immediately on server start
    asyncio.create_task(market_data_worker())

# --- API ENDPOINTS ---

@app.get("/stocks/all")
async def get_all_stocks(page: int = 1, limit: int = 50):
    """
    PAGINATED endpoint to handle thousands of stocks.
    """
    data = cache.get_all_stocks(page, limit)
    return {
        "page": page,
        "limit": limit,
        "total": len(cache.snapshot),
        "data": data,
        "last_updated": cache.last_updated
    }

@app.get("/stocks/{symbol}/history")
async def get_history(symbol: str, period: str = "1y", interval: str = "1d"):
    """
    Detailed historical data for charting.
    """
    ticker = f"{symbol}.NS"
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        raise HTTPException(status_code=404, detail="Stock not found")
    
    # Convert DataFrame to a list of dictionaries for the frontend
    df.reset_index(inplace=True)
    # Formatting date for JSON
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    return df.to_dict(orient="records")

@app.get("/stocks/{symbol}/fundamentals")
async def get_fundamentals(symbol: str):
    """
    Deep dive data: Market Cap, P/E, Dividend, etc.
    """
    ticker = f"{symbol}.NS"
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "symbol": symbol,
        "name": info.get('longName'),
        "marketCap": info.get('marketCap'),
        "peRatio": info.get('trailingPE'),
        "divYield": info.get('dividendYield'),
        "fiftyTwoWeekHigh": info.get('fiftyTwoWeekHigh'),
        "fiftyTwoWeekLow": info.get('fiftyTwoWeekLow'),
        "beta": info.get('beta'),
        "eps": info.get('trailingEps'),
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # CRITICAL FIX: Instead of calling yfinance, 
            # we simply send the current state of the CACHE.
            # This is nearly instant and zero-risk for rate limiting.
            await websocket.send_json({
                "type": "MARKET_UPDATE",
                "data": cache.snapshot, 
                "timestamp": cache.last_updated.isoformat() if cache.last_updated else None
            })
            await asyncio.sleep(10) # Send updates every 10 seconds
    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/")
def root():
    return {"status": "Running", "cache_size": len(cache.snapshot)}
