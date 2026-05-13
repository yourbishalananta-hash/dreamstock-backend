from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import datetime
import asyncio
import logging
import math
from concurrent.futures import ThreadPoolExecutor

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("marketpulse")

# ---------- App ----------
app = FastAPI(title="MarketPulse Pro Enterprise API")

ALL_INDIAN_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS",
    "INFY.NS", "SBIN.NS", "LICI.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "HCLTECH.NS", "MARUTI.NS", "SUNPHARMA.NS", "ADANIENT.NS",
    "AXISBANK.NS", "KOTAKBANK.NS", "WIPRO.NS", "ASIANPAINT.NS", "TITAN.NS",
    "ULTRACEMCO.NS", "POWERGRID.NS", "NTPC.NS", "NESTLEIND.NS", "TATAMOTORS.NS",
    "TATASTEEL.NS", "M&M.NS", "BAJAJFINSV.NS", "JSWSTEEL.NS",
]

INDIAN_INDICES = ["^NSEI", "^BSESN", "^NSEBANK"]

INDEX_DISPLAY_NAMES = {
    "^NSEI": "NIFTY 50",
    "^BSESN": "SENSEX",
    "^NSEBANK": "BANK NIFTY",
}


# ---------- Cache ----------
class MarketCache:
    def __init__(self):
        self.snapshot = {}
        self.indices = {}
        self.last_updated = None
        self.last_error = None
        self.success_count = 0
        self.failure_count = 0

    def update_snapshot(self, data):
        self.snapshot = data
        self.last_updated = datetime.now()

    def update_indices(self, data):
        self.indices = data

    def get_all_stocks(self, page: int, limit: int):
        symbols = list(self.snapshot.keys())
        start = (page - 1) * limit
        end = start + limit
        paginated = symbols[start:end]
        return [self.snapshot[s] for s in paginated]


cache = MarketCache()
executor = ThreadPoolExecutor(max_workers=8)

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Fetch logic ----------
def fetch_single_ticker(ticker: str):
    """Fetch one ticker's recent data. Returns dict or None."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="10d", interval="1d", auto_adjust=False)
        if df is None or df.empty:
            log.warning(f"  {ticker}: empty dataframe")
            return None

        close_series = df["Close"].dropna()
        if len(close_series) < 2:
            log.warning(f"  {ticker}: only {len(close_series)} valid close(s)")
            return None

        ltp = float(close_series.iloc[-1])
        base_price = float(close_series.iloc[-2])
        if math.isnan(ltp) or math.isnan(base_price) or math.isinf(ltp) or math.isinf(base_price):
            return None
        if base_price == 0:
            return None

        change = ((ltp - base_price) / base_price) * 100
        change_abs = ltp - base_price

        volume = 0
        if "Volume" in df.columns:
            vol_series = df["Volume"].dropna()
            if len(vol_series) > 0:
                v = vol_series.iloc[-1]
                if not math.isnan(v):
                    volume = int(v)

        turnover = round(ltp * volume, 2)
        display_symbol = ticker.replace(".NS", "") if ticker.endswith(".NS") else ticker

        return {
            "symbol": display_symbol,
            "ticker": ticker,
            "ltp": round(ltp, 2),
            "basePrice": round(base_price, 2),
            "change": round(change, 2),
            "changeAbs": round(change_abs, 2),
            "volume": volume,
            "turnover": turnover,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        log.warning(f"  {ticker}: {type(e).__name__}: {e}")
        return None


def fetch_all_market_data():
    log.info(f"Fetching {len(ALL_INDIAN_TICKERS)} tickers individually...")
    snapshot = {}
    success = 0
    failed = 0
    for ticker in ALL_INDIAN_TICKERS:
        result = fetch_single_ticker(ticker)
        if result:
            snapshot[ticker] = result
            success += 1
        else:
            failed += 1
    log.info(f"Stocks: {success} ok, {failed} failed")
    cache.success_count = success
    cache.failure_count = failed
    return snapshot


def fetch_all_indices():
    log.info(f"Fetching {len(INDIAN_INDICES)} indices...")
    snapshot = {}
    for idx in INDIAN_INDICES:
        result = fetch_single_ticker(idx)
        if result:
            result["symbol"] = INDEX_DISPLAY_NAMES.get(idx, idx)
            snapshot[idx] = result
    log.info(f"Indices: {len(snapshot)}/{len(INDIAN_INDICES)}")
    return snapshot


# ---------- Background worker ----------
async def market_data_worker():
    while True:
        log.info("Updating Global Market Cache...")
        try:
            loop = asyncio.get_event_loop()
            new_stocks = await loop.run_in_executor(executor, fetch_all_market_data)
            new_indices = await loop.run_in_executor(executor, fetch_all_indices)
            if new_stocks:
                cache.update_snapshot(new_stocks)
            if new_indices:
                cache.update_indices(new_indices)
            cache.last_error = None
            log.info(f"Cache updated. {len(new_stocks)} stocks + {len(new_indices)} indices.")
        except Exception as e:
            cache.last_error = f"{type(e).__name__}: {e}"
            log.exception("Worker iteration crashed")
        await asyncio.sleep(60)


@app.on_event("startup")
async def startup_event():
    log.info("Server starting; spawning market data worker.")
    asyncio.create_task(market_data_worker())


# ---------- Endpoints ----------
@app.get("/")
def root():
    return {
        "status": "Running",
        "cache_size": len(cache.snapshot),
        "indices_count": len(cache.indices),
        "last_updated": cache.last_updated.isoformat() if cache.last_updated else None,
        "last_error": cache.last_error,
        "last_success_count": cache.success_count,
        "last_failure_count": cache.failure_count,
    }


@app.get("/refresh")
async def manual_refresh():
    loop = asyncio.get_event_loop()
    try:
        new_stocks = await loop.run_in_executor(executor, fetch_all_market_data)
        new_indices = await loop.run_in_executor(executor, fetch_all_indices)
        if new_stocks:
            cache.update_snapshot(new_stocks)
        if new_indices:
            cache.update_indices(new_indices)
        cache.last_error = None
    except Exception as e:
        cache.last_error = f"{type(e).__name__}: {e}"
        log.exception("Manual refresh failed")
    return {
        "stocks": len(cache.snapshot),
        "indices": len(cache.indices),
        "last_updated": cache.last_updated.isoformat() if cache.last_updated else None,
        "last_error": cache.last_error,
    }


@app.get("/market/summary")
def market_summary():
    """Index data + summary stats for dashboard."""
    stocks = list(cache.snapshot.values())
    advancing = sum(1 for s in stocks if s["change"] > 0)
    declining = sum(1 for s in stocks if s["change"] < 0)
    unchanged = sum(1 for s in stocks if s["change"] == 0)

    return {
        "indices": list(cache.indices.values()),
        "totals": {
            "total": len(stocks),
            "advancing": advancing,
            "declining": declining,
            "unchanged": unchanged,
        },
        "last_updated": cache.last_updated.isoformat() if cache.last_updated else None,
    }


@app.get("/stocks/top/{category}")
def get_top(category: str, limit: int = 5):
    """
    category: gainers | losers | turnover | volume | active
    Returns top N stocks sorted by the relevant metric.
    """
    stocks = list(cache.snapshot.values())
    if not stocks:
        return {"category": category, "data": []}

    if category == "gainers":
        result = sorted([s for s in stocks if s["change"] > 0],
                        key=lambda s: s["change"], reverse=True)
    elif category == "losers":
        result = sorted([s for s in stocks if s["change"] < 0],
                        key=lambda s: s["change"])
    elif category == "turnover":
        result = sorted(stocks, key=lambda s: s.get("turnover", 0), reverse=True)
    elif category in ("volume", "active"):
        result = sorted(stocks, key=lambda s: s.get("volume", 0), reverse=True)
    else:
        raise HTTPException(status_code=400,
                            detail=f"Unknown category: {category}")

    return {"category": category, "data": result[:limit], "total": len(result)}


@app.get("/stocks/all")
async def get_all_stocks(page: int = 1, limit: int = 50):
    data = cache.get_all_stocks(page, limit)
    return {
        "page": page,
        "limit": limit,
        "total": len(cache.snapshot),
        "data": data,
        "last_updated": cache.last_updated,
    }


@app.get("/stocks/{symbol}/history")
async def get_history(symbol: str, period: str = "1y", interval: str = "1d"):
    ticker = symbol if ("." in symbol or symbol.startswith("^")) else f"{symbol}.NS"
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")

        df = df.reset_index()
        date_col = None
        for candidate in ("Date", "Datetime", "index"):
            if candidate in df.columns:
                date_col = candidate
                break
        if date_col is None:
            date_col = df.columns[0]
        df[date_col] = df[date_col].apply(
            lambda x: x.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(x, "strftime") else str(x)
        )
        if date_col != "Date":
            df = df.rename(columns={date_col: "Date"})
        df = df.dropna(subset=["Close"])
        return df.to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as e:
        log.exception(f"history fetch failed for {ticker}")
        raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}: {e}")


@app.get("/stocks/{symbol}/fundamentals")
async def get_fundamentals(symbol: str):
    ticker = symbol if ("." in symbol or symbol.startswith("^")) else f"{symbol}.NS"
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception as e:
        log.exception(f"fundamentals fetch failed for {ticker}")
        raise HTTPException(status_code=502, detail=f"upstream error: {type(e).__name__}: {e}")
    return {
        "symbol": symbol,
        "name": info.get("longName"),
        "marketCap": info.get("marketCap"),
        "peRatio": info.get("trailingPE"),
        "divYield": info.get("dividendYield"),
        "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
        "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
        "beta": info.get("beta"),
        "eps": info.get("trailingEps"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json({
                "type": "MARKET_UPDATE",
                "data": cache.snapshot,
                "indices": cache.indices,
                "timestamp": cache.last_updated.isoformat() if cache.last_updated else None,
            })
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
