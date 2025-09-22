from graph.pipeline import app

state = {"query": "Should i buy Infosys stocks?"}
final_state = app.invoke(state)

print("Index backend:", final_state["data_bundle"]["diagnostics"]["index_backend"])
print("\nTop evidence:")
for e in final_state["report"]["evidence_top3"]:
    print(f"- {e['title']}  |  {e['url']}")

print("\nAnalysis symbols:")
for sym, vals in final_state["report"]["analysis"]["symbols"].items():
    print(sym, "→", vals)

# Full structured report dict:
final_state["report"]
