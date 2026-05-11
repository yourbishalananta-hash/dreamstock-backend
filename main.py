from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta
import asyncio
import json

app = FastAPI(title="MarketPulse Pro API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# TECHNICAL INDICATORS
# ============================================

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
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

# ============================================
# STOCK DATABASE
# ============================================

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

TICKER_NAMES = {
    "RELIANCE": "Reliance Industries", "TCS": "Tata Consultancy Services",
    "HDFCBANK": "HDFC Bank", "ICICIBANK": "ICICI Bank",
    "BHARTIARTL": "Bharti Airtel", "INFY": "Infosys", "SBIN": "State Bank of India",
    "LICI": "LIC India", "ITC": "ITC Limited", "HINDUNILVR": "Hindustan Unilever",
    "LT": "Larsen & Toubro", "BAJFINANCE": "Bajaj Finance", "HCLTECH": "HCL Technologies",
    "MARUTI": "Maruti Suzuki", "SUNPHARMA": "Sun Pharmaceutical", "ADANIENT": "Adani Enterprises",
    "KOTAKBANK": "Kotak Mahindra Bank", "TITAN": "Titan Company", "ULTRACEMCO": "UltraTech Cement",
    "AXISBANK": "Axis Bank", "NTPC": "NTPC Limited", "ADANIPORTS": "Adani Ports",
    "ASIANPAINT": "Asian Paints", "COALINDIA": "Coal India", "BAJAJFINSV": "Bajaj Finserv",
    "BPCL": "Bharat Petroleum", "ONGC": "Oil & Natural Gas", "M&M": "Mahindra & Mahindra",
    "TATAMOTORS": "Tata Motors", "JSWSTEEL": "JSW Steel", "TATASTEEL": "Tata Steel",
    "ADANIPOWER": "Adani Power", "HINDALCO": "Hindalco Industries", "GRASIM": "Grasim Industries",
    "NESTLEIND": "Nestle India", "POWERGRID": "Power Grid Corp", "TECHM": "Tech Mahindra",
    "WIPRO": "Wipro Limited", "INDUSINDBK": "IndusInd Bank", "SBILIFE": "SBI Life Insurance",
}

# In-memory portfolio store (replace with DB in production)
_portfolio = {
    "balance": 1000000.0,
    "holdings": [],
    "transactions": []
}

# In-memory orders store
_orders = []

# ============================================
# HELPER: safe yfinance download
# ============================================

def download_single(ticker: str, period: str = "2mo", interval: str = "1d") -> pd.DataFrame:
    """Download a single ticker safely, compatible with yfinance >= 0.2."""
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
    # yfinance may return MultiIndex columns for a single ticker in newer versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


def download_batch(tickers: List[str], period: str = "2mo", interval: str = "1d"):
    """
    Download multiple tickers. Returns a dict of {ticker: DataFrame}.
    Compatible with yfinance >= 0.2 MultiIndex column structure.
    """
    raw = yf.download(
        tickers, period=period, interval=interval,
        group_by="ticker", auto_adjust=True, progress=False
    )
    result = {}
    for t in tickers:
        try:
            if len(tickers) == 1:
                df = raw.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            else:
                # MultiIndex: (field, ticker)
                if isinstance(raw.columns, pd.MultiIndex):
                    df = raw.xs(t, axis=1, level=1).copy()
                else:
                    df = raw[t].copy()
            result[t] = df.dropna()
        except Exception:
            result[t] = pd.DataFrame()
    return result

# ============================================
# MARKET WATCH
# ============================================

@app.get("/stocks/market-watch")
def get_market_data():
    """Get market watch data with RSI for all tracked stocks."""
    batch = download_batch(TICKERS, period="2mo", interval="1d")

    results = []
    for ticker in TICKERS:
        try:
            df = batch.get(ticker, pd.DataFrame())
            if df.empty or len(df) < 3:
                continue

            close = df["Close"].squeeze()
            rsi_series = calculate_rsi(close)

            current_price = float(close.iloc[-1])
            prev_price = float(close.iloc[-2])
            last_rsi = float(rsi_series.iloc[-1])
            change_pct = ((current_price - prev_price) / prev_price) * 100

            symbol = ticker.replace(".NS", "")
            results.append({
                "symbol": symbol,
                "name": TICKER_NAMES.get(symbol, symbol),
                "price": round(current_price, 2),
                "change": round(change_pct, 2),
                "rsi": round(last_rsi, 2) if not np.isnan(last_rsi) else 50.0,
                "status": (
                    "Overbought" if last_rsi > 70
                    else "Oversold" if last_rsi < 30
                    else "Neutral"
                ),
                "volume": int(df["Volume"].iloc[-1]),
                "high": round(float(df["High"].iloc[-1]), 2),
                "low": round(float(df["Low"].iloc[-1]), 2),
                "open": round(float(df["Open"].iloc[-1]), 2),
            })
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    return {"data": results, "timestamp": datetime.now().isoformat()}

# ============================================
# STOCK DETAILS
# ============================================

@app.get("/stocks/{symbol}/details")
def get_stock_details(symbol: str):
    """Get detailed fundamental information for a stock."""
    ticker_str = f"{symbol}.NS"
    try:
        stock = yf.Ticker(ticker_str)
        info = stock.info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch data: {e}")

    if not info or info.get("regularMarketPrice") is None and info.get("currentPrice") is None:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    return {
        "symbol": symbol,
        "name": info.get("longName", TICKER_NAMES.get(symbol, symbol)),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "marketCap": info.get("marketCap", 0),
        "peRatio": info.get("trailingPE") or 0,
        "eps": info.get("trailingEps") or 0,
        "bookValue": info.get("bookValue") or 0,
        "dividendYield": info.get("dividendYield") or 0,
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh") or 0,
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow") or 0,
        "beta": info.get("beta") or 0,
        "avgVolume": info.get("averageVolume") or 0,
        "description": info.get("longBusinessSummary", ""),
    }

# ============================================
# STOCK HISTORY  ← NEW
# ============================================

TIMEFRAME_MAP = {
    "1m":  ("1d",  "1m"),
    "5m":  ("5d",  "5m"),
    "15m": ("5d",  "15m"),
    "1H":  ("1mo", "1h"),
    "1D":  ("6mo", "1d"),
    "1W":  ("2y",  "1wk"),
    "1M":  ("5y",  "1mo"),
}

@app.get("/stocks/{symbol}/history")
def get_stock_history(symbol: str, timeframe: str = "1D", limit: int = 100):
    """Get OHLCV history for a stock. Used by the charts component."""
    period, interval = TIMEFRAME_MAP.get(timeframe, ("6mo", "1d"))
    ticker_str = f"{symbol}.NS"

    try:
        df = download_single(ticker_str, period=period, interval=interval)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No history for '{symbol}'")

    df = df.tail(limit)

    candles = []
    for ts, row in df.iterrows():
        candles.append({
            "time": int(ts.timestamp()),
            "open":  round(float(row["Open"]),   2),
            "high":  round(float(row["High"]),   2),
            "low":   round(float(row["Low"]),    2),
            "close": round(float(row["Close"]),  2),
            "volume": int(row["Volume"]),
        })

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "data": candles,
        "timestamp": datetime.now().isoformat(),
    }

# ============================================
# TECHNICAL INDICATORS
# ============================================

@app.get("/stocks/{symbol}/technicals")
def get_technicals(symbol: str):
    """Get all technical indicators for a stock."""
    ticker_str = f"{symbol}.NS"
    df = download_single(ticker_str, period="6mo", interval="1d")

    if df.empty:
        raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")

    close = df["Close"].squeeze()

    rsi          = calculate_rsi(close)
    macd, signal, histogram = calculate_macd(close)
    upper, middle, lower = calculate_bollinger_bands(close)

    def safe(val):
        v = float(val.iloc[-1]) if hasattr(val, "iloc") else float(val)
        return round(v, 2) if not np.isnan(v) else None

    return {
        "rsi":               safe(rsi),
        "macd":              safe(macd),
        "macdSignal":        safe(signal),
        "macdHistogram":     safe(histogram),
        "bollingerUpper":    safe(upper),
        "bollingerMiddle":   safe(middle),
        "bollingerLower":    safe(lower),
        "sma20":             safe(close.rolling(20).mean()),
        "sma50":             safe(close.rolling(50).mean()),
        "sma200":            safe(close.rolling(200).mean()),
        "ema12":             safe(close.ewm(span=12, adjust=False).mean()),
        "ema26":             safe(close.ewm(span=26, adjust=False).mean()),
        "timestamp":         datetime.now().isoformat(),
    }

# ============================================
# SCREENER  ← NEW
# ============================================

@app.post("/screener/search")
def screener_search(filters: dict):
    """
    Screen stocks by filter criteria.
    Supported filters: minPrice, maxPrice, minRsi, maxRsi,
                       minChange, maxChange, status
    """
    market = get_market_data()
    stocks = market["data"]

    min_price  = filters.get("minPrice",  0)
    max_price  = filters.get("maxPrice",  float("inf"))
    min_rsi    = filters.get("minRsi",    0)
    max_rsi    = filters.get("maxRsi",    100)
    min_change = filters.get("minChange", float("-inf"))
    max_change = filters.get("maxChange", float("inf"))
    status     = filters.get("status",    None)   # "Overbought" | "Oversold" | "Neutral"

    results = [
        s for s in stocks
        if (min_price  <= s["price"]  <= max_price)
        and (min_rsi   <= s["rsi"]    <= max_rsi)
        and (min_change <= s["change"] <= max_change)
        and (status is None or s["status"] == status)
    ]

    # Sort
    sort_by  = filters.get("sortBy",    "change")
    sort_dir = filters.get("sortOrder", "desc")
    results.sort(
        key=lambda x: x.get(sort_by, 0),
        reverse=(sort_dir == "desc")
    )

    limit = filters.get("limit", 50)
    return {
        "data":      results[:limit],
        "total":     len(results),
        "timestamp": datetime.now().isoformat(),
    }

# ============================================
# PORTFOLIO  ← NEW
# ============================================

@app.get("/portfolio")
def get_portfolio():
    """Return current portfolio with live P&L."""
    holdings_with_pnl = []
    total_invested = 0.0
    total_current  = 0.0

    for holding in _portfolio["holdings"]:
        symbol = holding["symbol"]
        try:
            df = download_single(f"{symbol}.NS", period="5d", interval="1d")
            ltp = round(float(df["Close"].iloc[-1]), 2) if not df.empty else holding["avgPrice"]
        except Exception:
            ltp = holding["avgPrice"]

        qty    = holding["quantity"]
        avg    = holding["avgPrice"]
        invest = qty * avg
        curr   = qty * ltp
        pnl    = curr - invest
        pnl_pct = (pnl / invest * 100) if invest else 0

        total_invested += invest
        total_current  += curr

        holdings_with_pnl.append({
            **holding,
            "ltp":       ltp,
            "invested":  round(invest, 2),
            "current":   round(curr,   2),
            "pnl":       round(pnl,    2),
            "pnlPct":    round(pnl_pct, 2),
        })

    total_pnl = total_current - total_invested

    return {
        "balance":         round(_portfolio["balance"], 2),
        "holdings":        holdings_with_pnl,
        "transactions":    _portfolio["transactions"][-50:],   # last 50
        "totalInvested":   round(total_invested, 2),
        "totalCurrent":    round(total_current,  2),
        "totalPnl":        round(total_pnl,      2),
        "totalPnlPct":     round((total_pnl / total_invested * 100) if total_invested else 0, 2),
        "timestamp":       datetime.now().isoformat(),
    }

# ============================================
# ORDERS / EXECUTE  ← NEW
# ============================================

@app.post("/orders/execute")
def execute_order(order: dict):
    """
    Execute a paper-trade order.
    Body: { symbol, quantity, orderType ("BUY"|"SELL"), price (optional) }
    """
    symbol    = order.get("symbol", "").upper()
    quantity  = int(order.get("quantity", 0))
    order_type = order.get("orderType", "BUY").upper()

    if not symbol or quantity <= 0:
        raise HTTPException(status_code=400, detail="Invalid symbol or quantity")

    # Get live price
    try:
        df = download_single(f"{symbol}.NS", period="5d", interval="1d")
        price = round(float(df["Close"].iloc[-1]), 2) if not df.empty else float(order.get("price", 0))
    except Exception:
        price = float(order.get("price", 0))

    if price <= 0:
        raise HTTPException(status_code=400, detail="Could not determine price")

    total_cost = price * quantity

    if order_type == "BUY":
        if _portfolio["balance"] < total_cost:
            raise HTTPException(status_code=400, detail="Insufficient balance")

        _portfolio["balance"] -= total_cost

        # Update holdings
        existing = next((h for h in _portfolio["holdings"] if h["symbol"] == symbol), None)
        if existing:
            total_qty = existing["quantity"] + quantity
            existing["avgPrice"] = round(
                (existing["avgPrice"] * existing["quantity"] + price * quantity) / total_qty, 2
            )
            existing["quantity"] = total_qty
        else:
            _portfolio["holdings"].append({
                "symbol":   symbol,
                "name":     TICKER_NAMES.get(symbol, symbol),
                "quantity": quantity,
                "avgPrice": price,
            })

    elif order_type == "SELL":
        existing = next((h for h in _portfolio["holdings"] if h["symbol"] == symbol), None)
        if not existing or existing["quantity"] < quantity:
            raise HTTPException(status_code=400, detail="Insufficient holdings")

        existing["quantity"] -= quantity
        _portfolio["balance"] += total_cost

        if existing["quantity"] == 0:
            _portfolio["holdings"].remove(existing)

    else:
        raise HTTPException(status_code=400, detail="orderType must be BUY or SELL")

    # Record transaction
    tx = {
        "id":        len(_portfolio["transactions"]) + 1,
        "symbol":    symbol,
        "quantity":  quantity,
        "price":     price,
        "total":     round(total_cost, 2),
        "type":      order_type,
        "timestamp": datetime.now().isoformat(),
    }
    _portfolio["transactions"].append(tx)
    _orders.append(tx)

    return {
        "success":   True,
        "order":     tx,
        "balance":   round(_portfolio["balance"], 2),
        "message":   f"{order_type} {quantity} {symbol} @ ₹{price}",
    }

# ============================================
# NEWS  ← NEW
# ============================================

NEWS_CATEGORIES = {
    "all":    None,
    "market": ["NIFTY", "SENSEX", "NSE", "BSE", "market"],
    "it":     ["TCS", "Infosys", "Wipro", "HCL", "Tech Mahindra"],
    "bank":   ["HDFC", "ICICI", "SBI", "Kotak", "Axis"],
    "energy": ["Reliance", "ONGC", "BPCL", "Adani"],
}

@app.get("/news")
def get_news(category: str = "all"):
    """
    Fetch news headlines via yfinance.
    Uses NIFTY 50 index ticker as the news source for market-wide news.
    """
    try:
        if category == "all" or category not in NEWS_CATEGORIES:
            ticker_str = "^NSEI"          # NIFTY 50 index
        else:
            symbols = NEWS_CATEGORIES[category]
            # Pick first matching ticker
            ticker_str = f"{symbols[0]}.NS" if symbols else "^NSEI"

        stock = yf.Ticker(ticker_str)
        raw_news = stock.news or []

        articles = []
        for item in raw_news[:20]:
            ct = item.get("content", {})
            pub_date = ct.get("pubDate", "") or ""
            # Format timestamp
            try:
                dt = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                formatted = dt.strftime("%d %b %Y, %I:%M %p")
            except Exception:
                formatted = pub_date

            articles.append({
                "id":          item.get("id", ""),
                "title":       ct.get("title", "No title"),
                "summary":     ct.get("summary", ""),
                "publisher":   ct.get("provider", {}).get("displayName", ""),
                "url":         ct.get("canonicalUrl", {}).get("url", ""),
                "publishedAt": formatted,
                "thumbnail":   (ct.get("thumbnail") or {}).get("resolutions", [{}])[0].get("url", ""),
                "category":    category,
            })

        return {
            "data":      articles,
            "category":  category,
            "count":     len(articles),
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch news: {e}")

# ============================================
# WEBSOCKET  (real-time updates)
# ============================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected")
    try:
        while True:
            try:
                data = get_market_data()
                await websocket.send_json(data)
            except Exception as e:
                print(f"WebSocket send error: {e}")
            await asyncio.sleep(10)          # push every 10 s (was 5 — reduce yfinance load)
    except WebSocketDisconnect:
        print("WebSocket client disconnected")

# ============================================
# ROOT
# ============================================

@app.get("/")
def root():
    return {
        "message":   "MarketPulse Pro API",
        "version":   "2.0.0",
        "endpoints": [
            "GET  /stocks/market-watch",
            "GET  /stocks/{symbol}/details",
            "GET  /stocks/{symbol}/history?timeframe=1D&limit=100",
            "GET  /stocks/{symbol}/technicals",
            "POST /screener/search",
            "GET  /portfolio",
            "POST /orders/execute",
            "GET  /news?category=all",
            "WS   /ws",
        ],
    }
