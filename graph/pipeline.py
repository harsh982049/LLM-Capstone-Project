from __future__ import annotations
from typing import TypedDict, Dict, Any
from langgraph.graph import StateGraph, END

from config import NSE_TICKERS, RSS_QUERIES
from agents.data_agent import DataAgent
from agents.analyst_agent import analyze
from agents.thesis_agent import generate_thesis
# We will add the verification agent back after this is fixed
from agents.verification_agent import verify

# --- THIS IS THE CRITICAL FIX ---
# We must define all the keys that will be added to the state during the graph's execution.
class State(TypedDict, total=False):
    query: str
    data_bundle: Dict[str, Any]
    analysis: Dict[str, Any]
    thesis: Dict[str, Any]         # <-- This key was missing, causing the error.
    verification: Dict[str, Any] # <-- We'll add this back later
    report: Dict[str, Any]
# --------------------------------

data_agent = DataAgent()

def node_data(state: State) -> State:
    q = state["query"]
    bundle = data_agent.run_pipeline(q, tickers=NSE_TICKERS, rss_queries=RSS_QUERIES, k=5)
    return {**state, "data_bundle": bundle}

def node_analyst(state: State) -> State:
    bundle = state["data_bundle"]
    analysis = analyze(bundle)
    return {**state, "analysis": analysis}

def node_thesis(state: State) -> State:
    bundle = state["data_bundle"]
    analysis = state["analysis"]
    thesis_result = generate_thesis(bundle, analysis)
    return {**state, "thesis": thesis_result}

def node_verification(state: State) -> State:
    analysis = state["analysis"]
    thesis = state["thesis"]
    verification_report = verify(analysis, thesis)
    return {**state, "verification": verification_report}

def node_output(state: State) -> State:
    # Now that the state is passed correctly, this will find the 'thesis' key.
    thesis = state.get("thesis", {"error": "thesis step failed or data was lost in pipeline"})
    verification = state.get("verification", {"error": "verification step did not run"})
    
    out = {
        "query": state["data_bundle"]["query"],
        "evidence_top3": [
            {k: v for k, v in e.items() if k in ("score","title","url","domain","published")}
            for e in state["data_bundle"]["evidence"][:3]
        ],
        "analysis": state["analysis"]["analysis"],
        "thesis": {
            "bull": thesis.get("thesis_bull", ""),
            "bear": thesis.get("thesis_bear", ""),
            "verdict_scaffold": thesis.get("verdict_scaffold", {}),
            "error": thesis.get("error")
        },
        "verification": verification
    }
    return {**state, "report": out}

# Build the simplified graph for now: data -> analyst -> thesis -> output
g = StateGraph(State)
g.add_node("data", node_data)
g.add_node("analyst", node_analyst)
g.add_node("thesis", node_thesis)
g.add_node("verification", node_verification)
g.add_node("output", node_output)

g.set_entry_point("data")

g.add_edge("data", "analyst")
g.add_edge("analyst", "thesis")
# g.add_edge("thesis", "output") # Temporarily bypass verification
g.add_edge("thesis", "verification")
g.add_edge("verification", "output")
g.add_edge("output", END)

app = g.compile()
