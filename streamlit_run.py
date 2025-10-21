import streamlit as st
import json
from graph.pipeline import app  # Import your main LangGraph app
from agents.final_summary import generate_final_summary

# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-Agent Investment Analyst",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Multi-Agent Investment Analyst for the Indian Market")
st.markdown("This app uses a team of AI agents to research and analyze stocks based on your questions. This is for educational purposes only and is not financial advice.")

# --- User Input ---
st.sidebar.header("Query")
user_query = st.sidebar.text_input("Ask a question about a major Indian stock (e.g., 'Should I buy Infosys today?')", "What is the outlook for Reliance Industries?")

if st.sidebar.button("Analyze Stock"):
    if not user_query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("🚀 The AI agents are collaborating... This may take a minute."):
            try:
                # --- Run the multi-agent pipeline ---
                initial_state = {"query": user_query}
                final_report = app.invoke(initial_state)['report']
                
                # --- Generate the final conversational summary ---
                with st.spinner("🤖 Synthesizing the final analysis..."):
                    final_summary = generate_final_summary(final_report)

                # --- Display the final, user-friendly response ---
                st.subheader("Your AI-Powered Analysis")
                st.markdown(f"### {final_summary.headline}")
                st.markdown(final_summary.summary)

                st.markdown("#### Key Takeaways:")
                for point in final_summary.key_points:
                    st.markdown(f"- {point}")
                
                st.info(f"**Next Steps:** {final_summary.next_steps}")

                # --- Display the detailed, raw output in an expander for transparency ---
                with st.expander("🔬 View Detailed Agent Output (JSON)"):
                    st.json(final_report)

            except Exception as e:
                st.error(f"An error occurred during the analysis: {e}")
                st.exception(e)

else:
    st.info("Enter a query on the left and click 'Analyze Stock' to begin.")
