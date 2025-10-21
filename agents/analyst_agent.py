from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np
import yfinance as yf
import time

def _series_from_timeseries(rows):
    df = pd.DataFrame(rows)
    if df.empty: return None
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date")
    return df

def analyze(bundle: Dict[str, Any]) -> Dict[str, Any]:
    market = bundle.get("market", {})
    ts = market.get("timeseries", {})
    results = {"symbols": {}}

    all_symbols = list(ts.keys())
    
    for sym in all_symbols:
        rows = ts.get(sym, [])
        df = _series_from_timeseries(rows)
        if df is None or "close" not in df.columns: continue

        ret = df["close"].pct_change().dropna()
        vol_10d = float(ret.tail(10).std() * np.sqrt(252)) if len(ret) >= 10 else None
        vol_20d = float(ret.tail(20).std() * np.sqrt(252)) if len(ret) >= 20 else None
        r_10d = float((df["close"].iloc[-1] / df["close"].iloc[-11] - 1)) if len(df) >= 11 else None
        r_20d = float((df["close"].iloc[-1] / df["close"].iloc[-21] - 1)) if len(df) >= 21 else None
        
        cummax = df["close"].cummax()
        drawdown = (df["close"] / cummax - 1.0).min()

        # --- Fetch Fundamental Ratios ---
        pe_ratio = None
        pb_ratio = None
        try:
            ticker_info = yf.Ticker(sym).info
            pe_ratio = ticker_info.get("trailingPE")
            pb_ratio = ticker_info.get("priceToBook")
            time.sleep(0.1) # Be respectful to the API
        except Exception:
            pass
        # ------------------------------------

        results["symbols"][sym] = {
            "latest_close": float(df["close"].iloc[-1]),
            "ret_10d": r_10d, "ret_20d": r_20d,
            "vol_10d": vol_10d, "vol_20d": vol_20d,
            "max_drawdown": float(drawdown),
            "pe": pe_ratio,
            "pb": pb_ratio
        }
    
    if "^NSEI" in ts:
        nifty = _series_from_timeseries(ts["^NSEI"])
        if nifty is not None:
            for sym, rows in ts.items():
                if sym == "^NSEI" or not results["symbols"].get(sym): continue
                df = _series_from_timeseries(rows)
                if df is None: continue
                merged = pd.merge(
                    df[["date","close"]].rename(columns={"close":"c_s"}),
                    nifty[["date","close"]].rename(columns={"close":"c_m"}),
                    on="date", how="inner"
                )
                if len(merged) < 20: continue
                sr = merged["c_s"].pct_change().dropna()
                mr = merged["c_m"].pct_change().dropna()
                L = min(len(sr), len(mr))
                if L >= 20:
                    sr, mr = sr.tail(L), mr.tail(L)
                    cov = np.cov(sr, mr)[0,1]; varm = np.var(mr)
                    beta = float(cov / varm) if varm > 0 else None
                    corr = float(sr.corr(mr))
                    results["symbols"][sym].update({"beta_vs_nifty": beta, "corr_vs_nifty": corr})

    return {
        "query": bundle.get("query", {}),
        "analysis": results,
        "used_symbols": list(results["symbols"].keys())
    }

