import json
from graph.pipeline import app

state = {"query": "Should I buy Infosys stock today?"}

# Invokes the entire pipeline
final_state = app.invoke(state)


print("\n--- 🚀 FINAL REPORT 🚀 ---\n")
if 'report' in final_state:
    print(json.dumps(final_state['report'], indent=2))
else:
    print("Error: 'report' key not found in the final state.")
    print("\n--- Full Final State ---")
    print(final_state)
