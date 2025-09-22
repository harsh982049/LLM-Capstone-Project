from __future__ import annotations
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END

from config import NSE_TICKERS, RSS_QUERIES
from agents.data_agent import DataAgent
from agents.analyst_agent import analyze

class State(TypedDict, total=False):
    query: str
    data_bundle: Dict[str, Any]
    analysis: Dict[str, Any]
    report: Dict[str, Any]

data_agent = DataAgent()

def node_data(state: State) -> State:
    q = state["query"]
    bundle = data_agent.run_pipeline(q, tickers=NSE_TICKERS, rss_queries=RSS_QUERIES, k=10)
    return {**state, "data_bundle": bundle}

def node_analyst(state: State) -> State:
    bundle = state["data_bundle"]
    analysis = analyze(bundle)
    return {**state, "analysis": analysis}

def node_output(state: State) -> State:
    # you can format richer output later; keep it simple for now
    out = {
        "query": state["data_bundle"]["query"],
        "evidence_top3": [
            {k: v for k, v in e.items() if k in ("score","title","url","domain","published")}
            for e in state["data_bundle"]["evidence"][:3]
        ],
        "analysis": state["analysis"]["analysis"]
    }
    return {**state, "report": out}

# Build the graph: query -> data -> analyst -> output -> END
g = StateGraph(State)
g.add_node("data", node_data)
g.add_node("analyst", node_analyst)
g.add_node("output", node_output)
g.set_entry_point("data")
g.add_edge("data", "analyst")
g.add_edge("analyst", "output")
g.add_edge("output", END)
app = g.compile()
