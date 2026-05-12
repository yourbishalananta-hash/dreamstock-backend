from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import datetime
import asyncio
import logging
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
    # add more as needed
]


# ---------- Cache ----------
class MarketCache:
    def __init__(self):
        self.snapshot = {}
        self.last_updated = None
        self.last_error = None
        self.success_count = 0
        self.failure_count = 0

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
    """
    Fetch one ticker's recent data. Returns a dict or None.
    Drops NaN rows so holidays/halts don't pollute the snapshot.
    """
    import math
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="10d", interval="1d", auto_adjust=False)
        if df is None or df.empty:
            log.warning(f"  {ticker}: empty dataframe")
            return None

        # Drop rows where Close is NaN — holidays, halts, missing data
        close_series = df["Close"].dropna()
        if len(close_series) < 2:
            log.warning(f"  {ticker}: only {len(close_series)} valid close(s) — need 2")
            return None

        ltp = float(close_series.iloc[-1])
        base_price = float(close_series.iloc[-2])

        # Belt-and-suspenders: reject anything that's still not a real number
        if math.isnan(ltp) or math.isnan(base_price) or math.isinf(ltp) or math.isinf(base_price):
            log.warning(f"  {ticker}: non-finite price after dropna (ltp={ltp}, base={base_price})")
            return None
        if base_price == 0:
            log.warning(f"  {ticker}: base price is zero, can't compute change")
            return None

        change = ((ltp - base_price) / base_price) * 100

        # Volume can also be NaN
        volume = 0
        if "Volume" in df.columns:
            vol_series = df["Volume"].dropna()
            if len(vol_series) > 0:
                v = vol_series.iloc[-1]
                if not math.isnan(v):
                    volume = int(v)

        return {
            "symbol": ticker.replace(".NS", ""),
            "ltp": round(ltp, 2),
            "basePrice": round(base_price, 2),
            "change": round(change, 2),
            "volume": volume,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        log.warning(f"  {ticker}: {type(e).__name__}: {e}")
        return None

        ltp = float(df["Close"].iloc[-1])
        base_price = float(df["Close"].iloc[-2])
        change = ((ltp - base_price) / base_price) * 100 if base_price else 0.0
        volume = int(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0

        return {
            "symbol": ticker.replace(".NS", ""),
            "ltp": round(ltp, 2),
            "basePrice": round(base_price, 2),
            "change": round(change, 2),
            "volume": volume,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        log.warning(f"  {ticker}: {type(e).__name__}: {e}")
        return None


def fetch_all_market_data():
    """Fetch all configured tickers one-by-one. Returns a dict keyed by ticker."""
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
    log.info(f"Fetch complete: {success} ok, {failed} failed")
    cache.success_count = success
    cache.failure_count = failed
    return snapshot


# ---------- Background worker ----------
async def market_data_worker():
    """Refresh cache every 60 seconds."""
    while True:
        log.info("Updating Global Market Cache...")
        try:
            loop = asyncio.get_event_loop()
            new_data = await loop.run_in_executor(executor, fetch_all_market_data)
            if new_data:
                cache.update_snapshot(new_data)
                cache.last_error = None
                log.info(f"Cache updated successfully. {len(new_data)} symbols cached.")
            else:
                cache.last_error = "fetch returned empty result"
                log.warning("Worker fetch returned empty. Cache unchanged.")
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
        "last_updated": cache.last_updated.isoformat() if cache.last_updated else None,
        "last_error": cache.last_error,
        "last_success_count": cache.success_count,
        "last_failure_count": cache.failure_count,
    }


@app.get("/refresh")
async def manual_refresh():
    """Trigger an immediate fetch and return the result. Useful for debugging."""
    loop = asyncio.get_event_loop()
    try:
        new_data = await loop.run_in_executor(executor, fetch_all_market_data)
        if new_data:
            cache.update_snapshot(new_data)
            cache.last_error = None
    except Exception as e:
        cache.last_error = f"{type(e).__name__}: {e}"
        log.exception("Manual refresh failed")
    return {
        "fetched": len(cache.snapshot),
        "cache_size": len(cache.snapshot),
        "last_updated": cache.last_updated.isoformat() if cache.last_updated else None,
        "last_error": cache.last_error,
        "success_count": cache.success_count,
        "failure_count": cache.failure_count,
    }


@app.get("/debug/test/{ticker}")
async def debug_single(ticker: str):
    """Test fetching a single ticker (use .NS suffix or it'll be added)."""
    loop = asyncio.get_event_loop()
    full_ticker = ticker if "." in ticker else f"{ticker}.NS"
    result = await loop.run_in_executor(executor, fetch_single_ticker, full_ticker)
    return {
        "ticker": full_ticker,
        "data": result,
        "success": result is not None,
    }


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
    ticker = symbol if "." in symbol else f"{symbol}.NS"
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
    except Exception as e:
        log.exception(f"history fetch failed for {ticker}")
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")
    if df.empty:
        raise HTTPException(status_code=404, detail="Stock not found")
    df.reset_index(inplace=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df.to_dict(orient="records")


@app.get("/stocks/{symbol}/fundamentals")
async def get_fundamentals(symbol: str):
    ticker = symbol if "." in symbol else f"{symbol}.NS"
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
    except Exception as e:
        log.exception(f"fundamentals fetch failed for {ticker}")
        raise HTTPException(status_code=502, detail=f"upstream error: {e}")
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
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(
                {
                    "type": "MARKET_UPDATE",
                    "data": cache.snapshot,
                    "timestamp": cache.last_updated.isoformat() if cache.last_updated else None,
                }
            )
            await asyncio.sleep(10)
    except WebSocketDisconnect:
        log.info("WebSocket client disconnected")
