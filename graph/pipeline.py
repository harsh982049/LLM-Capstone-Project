from __future__ import annotations
from typing import TypedDict, Dict, Any, Literal
import logging

from langgraph.graph import StateGraph, END

# Imports configurations and agents
from config import NSE_TICKERS, RSS_QUERIES
from agents.data_agent import DataAgent
from agents.analyst_agent import analyze
from agents.thesis_agent import generate_thesis
from agents.verification_agent import verify
from agents.router_agent import classify_intent 
from agents.portfolio_agent import suggest_portfolio 
from agents.simulation_agent import run_simulation 

log = logging.getLogger(__name__)

# Defines the State
class State(TypedDict, total=False):
    query: str                   # The initial user query
    intent: str                  # Classified intent from the router
    entities: list[str]          # Entities extracted by the router
    data_bundle: Dict[str, Any]  # Output from Data Agent
    analysis: Dict[str, Any]     # Output from Analyst Agent
    thesis: Dict[str, Any]       # Output from Thesis Agent
    verification: Dict[str, Any] # Output from Verification Agent
    portfolio_suggestion: Dict[str, Any]
    simulation_result: Dict[str, Any]
    report: Dict[str, Any] # The final consolidated report
    diagnostics: Dict[str, Any] # Diagnostics collected across steps

data_agent = DataAgent()

# Define Agent Nodes
def node_router(state: State) -> State:
    log.info("Executing Router Node")
    classification = classify_intent(state)
    current_diag = state.get("diagnostics", {})
    new_diag = classification.pop("diagnostics", {})
    return {**state, **classification, "diagnostics": {**current_diag, **new_diag}}

def node_data(state: State) -> State:
    log.info("Executing Data Node")
    q = state["query"]
    entities = state.get("entities", [])
    bundle = data_agent.run_pipeline(q, entities, default_tickers=NSE_TICKERS, rss_queries=RSS_QUERIES, k=5)
    current_diag = state.get("diagnostics", {})
    new_diag = bundle.pop("diagnostics", {})
    return {**state, "data_bundle": bundle, "diagnostics": {**current_diag, **new_diag}}

def node_analyst(state: State) -> State:
    log.info("Executing Analyst Node")
    if "data_bundle" not in state:
        log.warning("Analyst node skipped: data_bundle not found in state.")
        return state
    bundle = state["data_bundle"]
    analysis_result = analyze(bundle)
    current_diag = state.get("diagnostics", {})
    new_diag = analysis_result.pop("diagnostics", {})
    return {**state, "analysis": analysis_result, "diagnostics": {**current_diag, **new_diag}}

def node_thesis(state: State) -> State:
    log.info("Executing Thesis Node")
    if "data_bundle" not in state or "analysis" not in state:
        log.warning("Thesis node skipped: data_bundle or analysis not found in state.")
        return {**state, "thesis": {"error": "Prerequisite data (data_bundle or analysis) missing."}}
    bundle = state["data_bundle"]
    analysis = state["analysis"]
    thesis_result = generate_thesis(bundle, analysis)
    current_diag = state.get("diagnostics", {})
    # meta = thesis_result.pop("meta", {}) # Assuming meta contains diagnostics
    return {**state, "thesis": thesis_result, "diagnostics": current_diag} 

def node_verification(state: State) -> State:
    log.info("Executing Verification Node")
    if "analysis" not in state or "thesis" not in state or state["thesis"].get("error"):
        log.warning("Verification node skipped: analysis/thesis missing or thesis had error.")
        return {**state, "verification": {"error": "Prerequisite data missing or invalid."}}
    analysis = state["analysis"]
    thesis = state["thesis"]
    verification_report = verify(analysis, thesis)
    current_diag = state.get("diagnostics", {})
    return {**state, "verification": verification_report, "diagnostics": current_diag}

def node_portfolio(state: State) -> State:
    log.info("Executing Portfolio Node (Placeholder)")
    result = suggest_portfolio(state)
    return {**state, **result}

def node_simulation(state: State) -> State:
    log.info("Executing Simulation Node (Placeholder)")
    result = run_simulation(state)
    return {**state, **result}

def node_output_standard(state: State) -> State:
    """Consolidates the report for the standard stock analysis/comparison path."""
    log.info("Executing Standard Output Node")
    if "data_bundle" not in state or "analysis" not in state or "thesis" not in state:
         log.error("Standard Output Node: Missing required state keys (data_bundle, analysis, thesis).")
         report = {
              "query": state.get("query_info", {"text": state.get("query", "N/A")}),
              "error": "Pipeline failed to produce all required analysis components.",
              "diagnostics": state.get("diagnostics", {})
         }
         return {**state, "report": report}

    thesis = state.get("thesis", {"error": "thesis data missing"})
    verification = state.get("verification", {"error": "verification step did not run or failed"})

    report = {
        "query": state["data_bundle"].get("query", {}),
        "analysis_type": state.get("intent", "stock_analysis"),
        "primary_symbol": state["data_bundle"].get("primary_symbol"),
        "evidence_topk": [
            {k: v for k, v in e.items() if k in ("id","score","title","url","domain","published")}
            for e in state["data_bundle"].get("evidence", [])
        ],
        "analysis": state["analysis"].get("analysis", {}),
        "thesis": {
            "bull": thesis.get("thesis_bull", ""),
            "bear": thesis.get("thesis_bear", ""),
            "verdict_scaffold": thesis.get("verdict_scaffold", {}),
            "error": thesis.get("error") 
        },
        "verification": verification,
        "diagnostics": state.get("diagnostics", {})
    }
    return {**state, "report": report}

def route_intent(state: State) -> Literal["analyse_stock", "compare_stock", "suggest_portfolio", "run_simulation", "general_qa_or_fallback"]:
    """Determines the next step based on classified intent."""
    intent = state.get("intent")
    log.info(f"Routing based on intent: {intent}")

    if intent == "stock_analysis":
        return "analyse_stock"
    elif intent == "stock_comparison":
         log.info("Routing stock_comparison to standard analysis path.")
         return "analyse_stock"
    elif intent == "portfolio_request":
        return "suggest_portfolio"
    elif intent == "hypothetical_scenario":
        return "run_simulation"
    else: 
        log.warning(f"Routing intent '{intent}' to fallback/standard path.")
        return "general_qa_or_fallback" 

# Builds the Graph
workflow = StateGraph(State)

# Adds nodes
workflow.add_node("router", node_router)
workflow.add_node("data", node_data)
workflow.add_node("analyst", node_analyst)
workflow.add_node("thesis", node_thesis)
workflow.add_node("verification", node_verification)
workflow.add_node("output_standard", node_output_standard)
workflow.add_node("portfolio", node_portfolio)       # Placeholder node
workflow.add_node("simulation", node_simulation)      # Placeholder node

# Sets entry point
workflow.set_entry_point("router")

# Defines edges and conditional routing
workflow.add_conditional_edges(
    "router",
    route_intent,
    {
        "analyse_stock": "data",
        "compare_stock": "data", 
        "suggest_portfolio": "portfolio", 
        "run_simulation": "simulation",  
        "general_qa_or_fallback": "data" 
         # Alternative fallback: workflow.add_node("output_general", ...); "general_qa_or_fallback": "output_general"
    }
)

# Standard analysis path
workflow.add_edge("data", "analyst")
workflow.add_edge("analyst", "thesis")
workflow.add_edge("thesis", "verification")
workflow.add_edge("verification", "output_standard")

# End points for all paths
workflow.add_edge("output_standard", END)
workflow.add_edge("portfolio", END) 
workflow.add_edge("simulation", END) 


# Compile the graph
app = workflow.compile()

log.info("LangGraph workflow compiled successfully.")

# Optional: Visualize the graph (requires optional installs)
# try:
#     from PIL import Image
#     img = app.get_graph().draw_mermaid_png()
#     with open("graph_visualization.png", "wb") as f:
#         f.write(img)
#     log.info("Graph visualization saved to graph_visualization.png")
# except Exception as e:
#     log.warning(f"Could not generate graph visualization: {e}")
