# backend_app.py
import time
import logging
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from graph.pipeline import app as langgraph_app # Import your compiled LangGraph app
from agents.final_summary import generate_final_summary
# Import Pydantic models for type checking if needed later
# from agents.final_summary import StandardSummaryResponse, PortfolioSummaryResponse, SimulationSummaryResponse, ErrorSummaryResponse, PlaceholderResponse

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Flask App Initialization ---
flask_api = Flask(__name__)
# Enable CORS for requests from your Next.js app's domain during development
# For production, restrict the origin more tightly
CORS(flask_api, resources={r"/api/*": {"origins": "*"}}) # Allow all origins for now

# --- API Endpoint ---
@flask_api.route('/api/analyze', methods=['POST'])
def analyze_query():
    """
    API endpoint to handle user queries.
    Expects JSON: {"query": "user's question"}
    Returns JSON containing the final summarized response and the raw report.
    """
    t0 = time.time()
    log.info("Received request to /api/analyze")
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            log.warning("Received invalid request data.")
            return jsonify({"error": "Invalid input. 'query' field is required."}), 400

        user_query = data['query']
        log.info(f"Processing query: {user_query}")

        # --- Run the multi-agent pipeline ---
        initial_state = {"query": user_query}
        # Note: LangGraph's invoke is synchronous
        final_result_state = langgraph_app.invoke(initial_state)

        if not final_result_state or 'report' not in final_result_state:
            log.error(f"Pipeline did not produce a 'report'. Final state: {final_result_state}")
            return jsonify({"error": "Analysis pipeline failed to produce a report."}), 500

        final_report = final_result_state['report']
        log.info("Pipeline execution successful. Generating final summary.")

        # --- Generate the final conversational summary ---
        # This step also involves an LLM call
        summary_object = generate_final_summary(final_report)

        # Convert Pydantic model to dict for JSON response
        summary_dict = summary_object.dict()

        log.info(f"Successfully processed query in {time.time() - t0:.2f} seconds.")
        # Return both the summary and the raw report
        return jsonify({
            "summary": summary_dict,
            "raw_report": final_report
        }), 200

    except Exception as e:
        log.error(f"Error during API processing: {e}", exc_info=True)
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

# --- Run the Flask App ---
if __name__ == '__main__':
    # Use waitress or gunicorn for production instead of Flask's development server
    # For development:
    log.info("Starting Flask development server...")
    # Make it accessible on your network (change host if needed)
    flask_api.run(host='0.0.0.0', port=5001, debug=False) # Use a port like 5001 to avoid conflicts
