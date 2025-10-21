from __future__ import annotations
from typing import Dict, Any, List
import json

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from agents.thesis_agent import _make_llm # Reuse the same Gemini LLM

class FinalResponse(BaseModel):
    """The final, conversational response to the user."""
    headline: str = Field(description="A clear, one-sentence headline answering the user's question.")
    summary: str = Field(description="A 3-4 sentence paragraph summarizing the key findings, written in a helpful, advisory tone.")
    key_points: List[str] = Field(description="A list of 3-5 bullet points covering the most important bullish and bearish factors.")
    next_steps: str = Field(description="A concluding sentence suggesting what the user might consider next, framed as educational advice.")

def generate_final_summary(report: Dict[str, Any]) -> FinalResponse:
    """
    Takes the entire analysis report and generates a final, human-readable summary.
    
    """
    
    # We will serialize the full report to give the LLM complete context
    report_str = json.dumps(report, indent=2, default=str)
    
    sys_prompt = """
    You are an AI financial assistant. Your role is to synthesize a complex JSON report into a clear, balanced, and easy-to-understand answer for an investor.
    - Analyze the entire report, including the evidence, quantitative analysis, the bull/bear theses, and the verification agent's findings.
    - Your tone should be helpful, objective, and educational.
    - You MUST NOT give direct financial advice. Use phrases like "investors might consider," "factors to watch are," or "the data suggests."
    - Your response must be in the form of a JSON object that matches the requested schema.
    """.strip()
    
    human_prompt = f"""
    Here is the full analysis report for my query "{report['query']['text']}":

    {report_str}

    Please synthesize this entire report into a final, conversational response for me.
    """.strip()

    try:
        llm = _make_llm()
        structured_llm = llm.with_structured_output(FinalResponse, method="json_mode")
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=human_prompt)
        ]
        
        final_summary = structured_llm.invoke(messages)
        return final_summary
        
    except Exception as e:
        # Fallback in case of an error
        return FinalResponse(
            headline="Could not generate a full summary due to an error.",
            summary=f"An error occurred during the final analysis step: {e}",
            key_points=["Please review the detailed JSON output to see the raw data."],
            next_steps="Try re-running the analysis or checking the application logs."
        )
