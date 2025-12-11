import streamlit as st
import json
import logging
import time
from graph.pipeline import app 
from agents.final_summary import generate_final_summary
from pydantic import BaseModel
try:
    from agents.final_summary import StandardSummaryResponse, PortfolioSummaryResponse, SimulationSummaryResponse, ErrorSummaryResponse
except ImportError:
    st.error("Could not import response types from final_summary agent.")
    class BaseModel: pass
    class StandardSummaryResponse(BaseModel): pass
    class PortfolioSummaryResponse(BaseModel): pass
    class SimulationSummaryResponse(BaseModel): pass
    class ErrorSummaryResponse(BaseModel): pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

st.set_page_config(
    page_title="Multi-Agent Investment Analyst Chat",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Investment Analyst Chat")
st.markdown("""
Ask questions about major Indian stocks, compare assets, or discuss market scenarios.
**Disclaimer:** Educational purposes only. Not financial advice. Data may have delays.
""")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you with your investment analysis today?"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            summary_obj = message["content"]["summary_obj"]
            final_report = message["content"].get("final_report") 

            st.markdown(f"#### {summary_obj.headline}")
            st.markdown(summary_obj.summary)

            if isinstance(summary_obj, StandardSummaryResponse):
                if summary_obj.key_points:
                    st.markdown("**Key Takeaways:**")
                    for point in summary_obj.key_points:
                        st.markdown(f"- {point}")
                st.info(f"**Next Steps:** {summary_obj.next_steps}")

            elif isinstance(summary_obj, (PortfolioSummaryResponse, SimulationSummaryResponse)):
                 if summary_obj.key_points:
                     st.markdown("**Key Considerations/Impacts:**")
                     for point in summary_obj.key_points:
                         st.markdown(f"- {point}")
                 st.info(f"**Next Steps:** {summary_obj.next_steps}")

            elif isinstance(summary_obj, PlaceholderResponse):
                 if hasattr(summary_obj, 'details') and summary_obj.details:
                      st.markdown("---")
                      st.markdown(summary_obj.details)

            st.warning(f"**Disclaimer:** {summary_obj.disclaimer}", icon="⚠️")

            if final_report:
                 with st.expander("🔬 View Detailed Agent Output (JSON)"):
                      st.json(final_report)

        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask about INFY, compare stocks, or ask 'what if'..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        error_message = None
        final_report_data = None
        summary_object = None

        try:
            message_placeholder.markdown("🤔 Thinking... Invoking AI agents...")
            time.sleep(0.5) 
            message_placeholder.markdown("📊 Gathering data and analyzing...")
            time.sleep(1.0) 

            log.info(f"Invoking pipeline with query: {prompt}")
            initial_state = {"query": prompt}
            final_result = app.invoke(initial_state) # Use invoke

            if final_result and ('report' in final_result):
                final_report_data = final_result['report']
                log.info("Pipeline execution successful. Generating final summary.")
                message_placeholder.markdown("📝 Synthesizing final analysis...")

                summary_object = generate_final_summary(final_report_data)

                if isinstance(summary_object, ErrorSummaryResponse):
                    log.error(f"Final summary generation returned an error: {summary_object.summary}")
                    error_message = summary_object.summary
                    summary_object = None 
                else:
                    log.info("Final summary generated successfully.")

            else:
                error_message = "The analysis pipeline did not produce a final report. The query might be too complex or outside the supported scope."
                log.error(f"Pipeline execution failed or did not produce a report. Final state: {final_result}")

        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            log.error(f"Error during Streamlit chat execution: {e}", exc_info=True)

        if error_message:
            message_placeholder.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
        elif summary_object:
             message_placeholder.empty()

             st.markdown(f"#### {summary_object.headline}")
             st.markdown(summary_object.summary)

             if isinstance(summary_object, StandardSummaryResponse):
                 if summary_object.key_points:
                     st.markdown("**Key Takeaways:**")
                     for point in summary_object.key_points:
                         st.markdown(f"- {point}")
                 st.info(f"**Next Steps:** {summary_object.next_steps}")

             elif isinstance(summary_object, (PortfolioSummaryResponse, SimulationSummaryResponse)):
                  if summary_object.key_points:
                      st.markdown("**Key Considerations/Impacts:**")
                      for point in summary_object.key_points:
                          st.markdown(f"- {point}")
                  st.info(f"**Next Steps:** {summary_object.next_steps}")

             elif isinstance(summary_object, PlaceholderResponse):
                  if hasattr(summary_object, 'details') and summary_object.details:
                       st.markdown("---")
                       st.markdown(summary_object.details)

             st.warning(f"**Disclaimer:** {summary_object.disclaimer}", icon="⚠️")

             if final_report_data:
                  with st.expander("🔬 View Detailed Agent Output (JSON)"):
                       st.json(final_report_data)

             st.session_state.messages.append({"role": "assistant", "content": {"summary_obj": summary_object, "final_report": final_report_data}})
        else:
             fallback_msg = "Sorry, I wasn't able to generate a response for that query."
             message_placeholder.markdown(fallback_msg)
             st.session_state.messages.append({"role": "assistant", "content": fallback_msg})