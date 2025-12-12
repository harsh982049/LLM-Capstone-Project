

from fastapi import FastAPI, Query
from typing import Dict, Any
import copy
import traceback

from graph.pipeline import app as workflow_app
from agents.final_summary import generate_final_summary

app = FastAPI(title="Pipeline-based Investment Analysis API")

@app.get("/full-report")
def full_report(query: str = Query(..., description="Text query (sentence/phrase)")):
    output: Dict[str, Any] = {"steps": [], "final_report": None, "summary": None}

    try:
        state: Dict[str, Any] = {"query": query, "diagnostics": {}}

        # Use the compiled workflow invoke method
        final_state = workflow_app.invoke(state)

        # Collect intermediate steps from diagnostics (if your agents already log)
        # Here we use state snapshots for simplicity
        for node_name, node_diag in final_state.get("diagnostics", {}).items():
            output["steps"].append({
                "step": node_name,
                "state_snapshot": node_diag
            })

        # Final report
        output["final_report"] = final_state.get("report")

        # Summary
        if output["final_report"]:
            try:
                summary_obj = generate_final_summary(output["final_report"])
                output["summary"] = summary_obj
            except Exception as e_summary:
                output["summary"] = {"error": f"Summary generation failed: {str(e_summary)}"}

        return output

    except Exception as e:
        return {
            "error": "Pipeline execution failed",
            "details": str(e),
            "trace": traceback.format_exc()
        }


