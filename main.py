from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
from datetime import datetime
import asyncio
import logging
import math
from concurrent.futures import ThreadPoolExecutor

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("marketpulse")

# ---------- App ----------
app = FastAPI(title="MarketPulse Pro Enterprise API")

# Master list of supported symbols with display info — used by autocomplete
SYMBOL_LIBRARY = [
    {"symbol": "RELIANCE",   "name": "Reliance Industries",        "ticker": "RELIANCE.NS"},
    {"symbol": "TCS",        "name": "Tata Consultancy Services",  "ticker": "TCS.NS"},
    {"symbol": "HDFCBANK",   "name": "HDFC Bank",                  "ticker": "HDFCBANK.NS"},
    {"symbol": "ICICIBANK",  "name": "ICICI Bank",                 "ticker": "ICICIBANK.NS"},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel",              "ticker": "BHARTIARTL.NS"},
    {"symbol": "INFY",       "name": "Infosys",                    "ticker": "INFY.NS"},
    {"symbol": "SBIN",       "name": "State Bank of India",        "ticker": "SBIN.NS"},
    {"symbol": "LICI",       "name": "Life Insurance Corp",        "ticker": "LICI.NS"},
    {"symbol": "ITC",        "name": "ITC Limited",                "ticker": "ITC.NS"},
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever",         "ticker": "HINDUNILVR.NS"},
    {"symbol": "LT",         "name": "Larsen & Toubro",            "ticker": "LT.NS"},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance",              "ticker": "BAJFINANCE.NS"},
    {"symbol": "HCLTECH",    "name": "HCL Technologies",           "ticker": "HCLTECH.NS"},
    {"symbol": "MARUTI",     "name": "Maruti Suzuki",              "ticker": "MARUTI.NS"},
    {"symbol": "SUNPHARMA",  "name": "Sun Pharmaceutical",         "ticker": "SUNPHARMA.NS"},
    {"symbol": "ADANIENT",   "name": "Adani Enterprises",          "ticker": "ADANIENT.NS"},
    {"symbol": "AXISBANK",   "name": "Axis Bank",                  "ticker": "AXISBANK.NS"},
    {"symbol": "KOTAKBANK",  "name": "Kotak Mahindra Bank",        "ticker": "KOTAKBANK.NS"},
    {"symbol": "WIPRO",      "name": "Wipro",                      "ticker": "WIPRO.NS"},
    {"symbol": "ASIANPAINT", "name": "Asian Paints",               "ticker": "ASIANPAINT.NS"},
    {"symbol": "TITAN",      "name": "Titan Company",              "ticker": "TITAN.NS"},
    {"symbol": "ULTRACEMCO", "name": "UltraTech Cement",           "ticker": "ULTRACEMCO.NS"},
    {"symbol": "POWERGRID",  "name": "Power Grid Corp",            "ticker": "POWERGRID.NS"},
    {"symbol": "NTPC",       "name": "NTPC Limited",               "ticker": "NTPC.NS"},
    {"symbol": "NESTLEIND",  "name": "Nestle India",               "ticker": "NESTLEIND.NS"},
    {"symbol": "TATAMOTORS", "name": "Tata Motors",                "ticker": "TATAMOTORS.NS"},
    {"symbol": "TATASTEEL",  "name": "Tata Steel",                 "ticker": "TATASTEEL.NS"},
    {"symbol": "M&M",        "name": "Mahindra & Mahindra",        "ticker": "M&M.NS"},
    {"symbol": "BAJAJFINSV", "name": "Bajaj Finserv",              "ticker": "BAJAJFINSV.NS"},
    {"symbol": "JSWSTEEL",   "name": "JSW Steel",                  "ticker": "JSWSTEEL.NS"},
    # Indices
    {"symbol": "NIFTY 50",   "name": "NIFTY 50 Index",             "ticker": "^NSEI",     "isIndex": True},
    {"symbol": "SENSEX",     "name": "BSE SENSEX",                 "ticker": "^BSESN",    "isIndex": True},
    {"symbol": "BANK NIFTY", "name": "Bank Nifty",                 "ticker": "^NSEBANK",  "isIndex": True},
]

ALL_INDIAN_TICKERS = [s["ticker"] for s in SYMBOL_LIBRARY if not s.get("isIndex")]
INDIAN_INDICES = [s["ticker"] for s in SYMBOL_LIBRARY if s.get("isIndex")]

INDEX_DISPLAY_NAMES = {s["ticker"]: s["symbol"] for s in SYMBOL_LIBRARY if s.get("isIndex")}
SYMBOL_TO_NAME = {s["symbol"]: s["name"] for s in SYMBOL_LIBRARY}


# ---------- Cache ----------
class MarketCache:
    def __init__(self):
        self.snapshot = {}        # ticker -> dict
        self.indices = {}
        self.fundamentals = {}    # symbol -> dict (refreshed less often)
        self.last_updated = None
        self.last_error = None
        self.success_count = 0
        self.failure_count = 0

    def get_all_stocks(self, page, limit):
        symbols = list(self.snapshot.keys())
        start = (page - 1) * limit
        return [self.snapshot[s] for s in symbols[start:start + limit]]


cache = MarketCache()
executor = ThreadPoolExecutor(max_workers=8)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- Fetch ----------
def fetch_single_ticker(ticker: str):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="10d", interval="1d", auto_adjust=False)
        if df is None or df.empty:
            return None

        close_series = df["Close"].dropna()
        if len(close_series) < 2:
            return None

        ltp = float(close_series.iloc[-1])
        base_price = float(close_series.iloc[-2])
        if not all(math.isfinite(x) for x in (ltp, base_price)) or base_price == 0:
            return None

        change = ((ltp - base_price) / base_price) * 100
        change_abs = ltp - base_price

        # Day high/low from the last available row
        last_row = df.iloc[-1]
        day_high = float(last_row.get("High", ltp)) if math.isfinite(float(last_row.get("High", ltp) or 0)) else ltp
        day_low  = float(last_row.get("Low",  ltp)) if math.isfinite(float(last_row.get("Low",  ltp) or 0)) else ltp
        day_open = float(last_row.get("Open", ltp)) if math.isfinite(float(last_row.get("Open", ltp) or 0)) else ltp

        volume = 0
        if "Volume" in df.columns:
            vol_series = df["Volume"].dropna()
            if len(vol_series) > 0:
                v = vol_series.iloc[-1]
                if math.isfinite(v):
                    volume = int(v)

        turnover = round(ltp * volume, 2)
        display_symbol = ticker.replace(".NS", "") if ticker.endswith(".NS") else ticker

        return {
            "symbol": display_symbol,
            "ticker": ticker,
            "name": SYMBOL_TO_NAME.get(display_symbol) or SYMBOL_TO_NAME.get(INDEX_DISPLAY_NAMES.get(ticker, ""), display_symbol),
            "ltp": round(ltp, 2),
            "basePrice": round(base_price, 2),
            "change": round(change, 2),
            "changeAbs": round(change_abs, 2),
            "dayHigh": round(day_high, 2),
            "dayLow":  round(day_low, 2),
            "dayOpen": round(day_open, 2),
            "volume": volume,
            "turnover": turnover,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        log.warning(f"  {ticker}: {type(e).__name__}: {e}")
        return None


def fetch_all_market_data():
    log.info(f"Fetching {len(ALL_INDIAN_TICKERS)} tickers...")
    snapshot = {}
    success = failed = 0
    for ticker in ALL_INDIAN_TICKERS:
        r = fetch_single_ticker(ticker)
        if r:
            snapshot[ticker] = r
            success += 1
        else:
            failed += 1
    log.info(f"Stocks: {success} ok, {failed} failed")
    cache.success_count = success
    cache.failure_count = failed
    return snapshot


def fetch_all_indices():
    snapshot = {}
    for idx in INDIAN_INDICES:
        r = fetch_single_ticker(idx)
        if r:
            r["symbol"] = INDEX_DISPLAY_NAMES.get(idx, idx)
            snapshot[idx] = r
    log.info(f"Indices: {len(snapshot)}/{len(INDIAN_INDICES)}")
    return snapshot


def fetch_fundamentals(ticker: str):
    """Fetch fundamentals + 52w high/low. Slower, so we cache it longer."""
    try:
        s = yf.Ticker(ticker)
        info = s.info or {}
        return {
            "longName":         info.get("longName"),
            "marketCap":        info.get("marketCap"),
            "peRatio":          info.get("trailingPE"),
            "pbRatio":          info.get("priceToBook"),
            "divYield":         info.get("dividendYield"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow":  info.get("fiftyTwoWeekLow"),
            "beta":             info.get("beta"),
            "eps":              info.get("trailingEps"),
            "sector":           info.get("sector"),
            "industry":         info.get("industry"),
            "sharesOutstanding":info.get("sharesOutstanding"),
            "bookValue":        info.get("bookValue"),
            "dividendRate":     info.get("dividendRate"),
            "currency":         info.get("currency", "INR"),
        }
    except Exception as e:
        log.warning(f"fundamentals fetch failed for {ticker}: {e}")
        return {}


# ---------- Worker ----------
async def market_data_worker():
    while True:
        log.info("Updating Global Market Cache...")
        try:
            loop = asyncio.get_event_loop()
            new_stocks = await loop.run_in_executor(executor, fetch_all_market_data)
            new_indices = await loop.run_in_executor(executor, fetch_all_indices)
            if new_stocks:
                cache.snapshot = new_stocks
                cache.last_updated = datetime.now()
            if new_indices:
                cache.indices = new_indices
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
            cache.snapshot = new_stocks
            cache.last_updated = datetime.now()
        if new_indices:
            cache.indices = new_indices
        cache.last_error = None
    except Exception as e:
        cache.last_error = f"{type(e).__name__}: {e}"
    return {"stocks": len(cache.snapshot), "indices": len(cache.indices), "last_error": cache.last_error}


@app.get("/stocks/symbols")
def get_symbol_library():
    """Returns the full searchable symbol list for autocomplete (works even
    when cache is empty)."""
    return {"symbols": SYMBOL_LIBRARY}


@app.get("/market/summary")
def market_summary():
    stocks = list(cache.snapshot.values())
    return {
        "indices": list(cache.indices.values()),
        "totals": {
            "total":      len(stocks),
            "advancing":  sum(1 for s in stocks if s["change"] > 0),
            "declining":  sum(1 for s in stocks if s["change"] < 0),
            "unchanged":  sum(1 for s in stocks if s["change"] == 0),
        },
        "last_updated": cache.last_updated.isoformat() if cache.last_updated else None,
    }


@app.get("/stocks/top/{category}")
def get_top(category: str, limit: int = 5):
    stocks = list(cache.snapshot.values())
    if not stocks:
        return {"category": category, "data": []}
    if category == "gainers":
        result = sorted([s for s in stocks if s["change"] > 0], key=lambda s: s["change"], reverse=True)
    elif category == "losers":
        result = sorted([s for s in stocks if s["change"] < 0], key=lambda s: s["change"])
    elif category == "turnover":
        result = sorted(stocks, key=lambda s: s.get("turnover", 0), reverse=True)
    elif category in ("volume", "active"):
        result = sorted(stocks, key=lambda s: s.get("volume", 0), reverse=True)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown: {category}")
    return {"category": category, "data": result[:limit], "total": len(result)}


@app.get("/stocks/all")
async def get_all_stocks(page: int = 1, limit: int = 50):
    return {
        "page": page,
        "limit": limit,
        "total": len(cache.snapshot),
        "data": cache.get_all_stocks(page, limit),
        "last_updated": cache.last_updated,
    }


def resolve_ticker(symbol: str) -> str:
    """Map a UI-facing symbol to the yfinance ticker."""
    s = symbol.strip()
    # Already a full ticker (RELIANCE.NS, ^NSEI)
    if "." in s or s.startswith("^"):
        return s
    # Friendly name
    for entry in SYMBOL_LIBRARY:
        if entry["symbol"].upper() == s.upper():
            return entry["ticker"]
    # Fallback: add .NS for NSE
    return f"{s}.NS"


@app.get("/stocks/{symbol}/history")
async def get_history(symbol: str, period: str = "1y", interval: str = "1d"):
    ticker = resolve_ticker(symbol)
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {ticker}")
        df = df.reset_index()
        date_col = next((c for c in ("Date", "Datetime", "index") if c in df.columns), df.columns[0])
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
async def get_fundamentals_endpoint(symbol: str):
    ticker = resolve_ticker(symbol)
    loop = asyncio.get_event_loop()
    info = await loop.run_in_executor(executor, fetch_fundamentals, ticker)
    return {"symbol": symbol, **info}


@app.get("/stocks/{symbol}/detail")
async def get_stock_detail(symbol: str):
    """One call returns snapshot + fundamentals + 1mo history. Used by detail page."""
    ticker = resolve_ticker(symbol)
    loop = asyncio.get_event_loop()

    # Snapshot (from cache if available, else fresh)
    snapshot = cache.snapshot.get(ticker) or cache.indices.get(ticker)
    if not snapshot:
        snapshot = await loop.run_in_executor(executor, fetch_single_ticker, ticker)

    # Fundamentals (always fresh; yfinance .info is fast enough)
    fundamentals = await loop.run_in_executor(executor, fetch_fundamentals, ticker)

    # History (1mo for the mini chart)
    history = []
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1mo", interval="1d", auto_adjust=False)
        if df is not None and not df.empty:
            df = df.reset_index()
            date_col = next((c for c in ("Date", "Datetime", "index") if c in df.columns), df.columns[0])
            df[date_col] = df[date_col].apply(
                lambda x: x.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(x, "strftime") else str(x)
            )
            if date_col != "Date":
                df = df.rename(columns={date_col: "Date"})
            df = df.dropna(subset=["Close"])
            history = df.to_dict(orient="records")
    except Exception as e:
        log.warning(f"history for detail failed for {ticker}: {e}")

    return {
        "symbol": symbol,
        "ticker": ticker,
        "snapshot": snapshot,
        "fundamentals": fundamentals,
        "history": history,
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
