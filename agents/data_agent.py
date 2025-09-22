from __future__ import annotations
import time, re, json
from typing import List, Dict, Any
import pandas as pd
import yfinance as yf
import feedparser
from transformers import AutoTokenizer
from pathlib import Path

from config import VECTOR_FAISS_DIR, EMBEDDING_MODEL, TOP_K, TIMEZONE
from agents.index_loader import RetrieverIndex

tok_cache = None
def _tok():
    global tok_cache
    if tok_cache is None:
        tok_cache = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return tok_cache

def _trim_text(txt: str, max_tokens: int) -> str:
    if not max_tokens: return txt
    ids = _tok().encode(txt, add_special_tokens=False)
    return txt if len(ids) <= max_tokens else _tok().decode(ids[:max_tokens], skip_special_tokens=True)

def _latest_summary(df: pd.DataFrame) -> dict:
    s = df.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.tz_localize(None)
    s = s.sort_values("date")
    last = s.iloc[-1]
    prev = s.iloc[-2] if len(s) > 1 else last
    closes = s["close"].astype(float).tail(10).pct_change().dropna()
    vol10 = float(closes.std() * (252 ** 0.5)) if len(closes) > 1 else None
    pct = float((last["close"] - prev["close"]) / prev["close"]) if prev["close"] else 0.0
    return {"as_of": last["date"].isoformat(), "latest_close": float(last["close"]), "pct_change_1d": pct, "realized_vol_10d": vol10, "vendor": "yfinance"}

class DataAgent:
    def __init__(self):
        self.index = RetrieverIndex(VECTOR_FAISS_DIR, EMBEDDING_MODEL)

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        return self.index.search(query, k=k)

    def fetch_prices(self, symbols: List[str], period="1mo", interval="1d") -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        for s in symbols:
            try:
                df = yf.Ticker(s).history(period=period, interval=interval, auto_adjust=False)
                if df.empty: continue
                df = df.rename_axis("date").reset_index()
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Adj Close":"adj_close","Volume":"volume"})
                if "adj_close" not in df.columns and "close" in df.columns:
                    df["adj_close"] = df["close"]
                out[s] = df[["date","open","high","low","close","adj_close","volume"]].to_dict(orient="records")
                time.sleep(0.15)
            except Exception:
                pass
        return out

    @staticmethod
    def _rss_url(q: str, hl="en-IN", gl="IN", ceid="IN:en") -> str:
        from urllib.parse import quote_plus
        return f"https://news.google.com/rss/search?q={quote_plus(q)}&hl={hl}&gl={gl}&ceid={ceid}"

    def fetch_rss(self, queries: List[str], per_query_limit=30, sleep_s=0.25) -> List[Dict[str, Any]]:
        items, seen = [], set()
        for q in queries:
            feed_url = self._rss_url(q)
            feed = feedparser.parse(feed_url)
            entries = feed.entries[:per_query_limit] if getattr(feed, "entries", None) else []
            for e in entries:
                link = (e.get("link") or "").strip()
                if not link or link in seen: continue
                seen.add(link)
                items.append({
                    "title": (e.get("title") or "").strip(),
                    "link": link,
                    "published": (e.get("published") or e.get("updated") or "").strip(),
                    "source_feed": feed_url,
                    "query": q
                })
            time.sleep(sleep_s)
        return items

    def run_pipeline(self, user_query: str, tickers: List[str], rss_queries: List[str], k: int = TOP_K, limit_tokens_for_evidence: int = 256) -> Dict[str, Any]:
        t0 = time.time()
        ev_raw = self.retrieve(user_query, k=k)
        evidence = [{
            "id": int(e.get("id", e.get("chunk", 0))),
            "external_id": f"{e.get('url','')}|{e.get('chunk',0)}",
            "url": e.get("url",""),
            "title": e.get("title",""),
            "published": e.get("published",""),
            "domain": e.get("domain",""),
            "score": float(e.get("score",0.0)),
            "chunk": int(e.get("chunk",0)),
            "text": _trim_text(e.get("text",""), limit_tokens_for_evidence)
        } for e in ev_raw]

        prices_raw = self.fetch_prices(tickers)
        market = {"symbols": {}, "timeseries": {}}
        for sym, rows in prices_raw.items():
            df = pd.DataFrame(rows)
            if not df.empty:
                market["symbols"][sym] = _latest_summary(df)
                market["timeseries"][sym] = df.tail(5).to_dict(orient="records")

        headlines = self.fetch_rss(rss_queries)

        bundle = {
            "query": {"text": user_query, "timestamp": pd.Timestamp.now(tz=TIMEZONE).isoformat()},
            "evidence": evidence,
            "market": market,
            "news": {"rss": headlines, "source": "GoogleNewsRSS"},
            "diagnostics": {
                "index_backend": self.index.backend,
                "vectors": len(getattr(self.index, "meta", [])),
                "timing_ms": int((time.time() - t0) * 1000)
            }
        }

        # Persist bundle (fixture for later agents)
        runs = Path("runs"); runs.mkdir(exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", user_query.lower()).strip("-")[:60]
        out_path = runs / f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{slug}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, ensure_ascii=False, indent=2, default=str)
        bundle["diagnostics"]["persisted"] = str(out_path)
        return bundle
