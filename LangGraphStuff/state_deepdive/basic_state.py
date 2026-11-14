from typing import TypedDict
from langgraph.graph import END, StateGraph


class SimplesState(TypedDict):
    count: str


def increment(state: SimplesState) -> SimplesState:
    return {"count": state["count"] + 1}


def should_continue(state):
    if state["count"] < 5:
        return "continue"
    else:
        return "stop"


graph = StateGraph(SimplesState)

graph.add_node("increment", increment)
graph.set_entry_point("increment")
graph.add_conditional_edges(
    "increment",
    should_continue,
    {"continue": "increment", "stop": END},
)

state = {"count": 0}

app = graph.compile()

result = app.invoke(state)

print(result)
