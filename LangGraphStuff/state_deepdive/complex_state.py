from typing import TypedDict, List, Annotated
from langgraph.graph import END, StateGraph
import operator


class SimplesState(TypedDict):
    count: str
    sum: Annotated[int, operator.add]
    # history: List[int]


def increment(state: SimplesState) -> SimplesState:
    return {"count": state["count"] + 1, "sum": state["count"] + 1}


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

state = {"count": 0, "sum": 0}

app = graph.compile()

result = app.invoke(state)

print(result)
