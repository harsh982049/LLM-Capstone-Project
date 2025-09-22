from __future__ import annotations
from typing import Dict, Any
import pandas as pd
import numpy as np

def _series_from_timeseries(rows):
    df = pd.DataFrame(rows)
    if df.empty: return None
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values("date")
    return df

def analyze(bundle: Dict[str, Any]) -> Dict[str, Any]:
    market = bundle.get("market", {})
    ts = market.get("timeseries", {})
    symbols = market.get("symbols", {})

    results = {"symbols": {}}

    # basic metrics per symbol
    for sym, rows in ts.items():
        df = _series_from_timeseries(rows)
        if df is None or "close" not in df.columns: 
            continue
        latest_close = float(df["close"].iloc[-1])
        ret = df["close"].pct_change().dropna()
        vol_10d = float(ret.tail(10).std() * np.sqrt(252)) if len(ret) >= 10 else None
        vol_5d = float(ret.tail(5).std() * np.sqrt(252)) if len(ret) >= 5 else None

        results["symbols"][sym] = {
            "latest_close": latest_close,
            "vol_10d": vol_10d,
            "vol_5d": vol_5d,
            # placeholders for fundamentals you may add
            "pe": None,
            "pb": None
        }

    # simple pairwise correlation example if at least two series exist
    if len(ts) >= 2:
        keys = list(ts.keys())[:2]
        df1 = _series_from_timeseries(ts[keys[0]])
        df2 = _series_from_timeseries(ts[keys[1]])
        if df1 is not None and df2 is not None:
            merged = pd.merge(
                df1[["date","close"]].rename(columns={"close":f"close_{keys[0]}"}),
                df2[["date","close"]].rename(columns={"close":f"close_{keys[1]}"}),
                on="date", how="inner"
            )
            if len(merged) >= 5:
                r = float(merged[f"close_{keys[0]}"].pct_change().dropna().corr(
                          merged[f"close_{keys[1]}"].pct_change().dropna()))
                results["pair_correlation"] = { "symbols": keys, "corr": r }

    return {
        "query": bundle.get("query", {}),
        "analysis": results,
        "used_symbols": list(results["symbols"].keys())
    }
