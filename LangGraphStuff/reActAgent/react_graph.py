"""Build and run a simple ReAct agent with LangGraph."""

from dotenv import load_dotenv

load_dotenv()

from langchain_core.agents import AgentFinish
from langgraph.graph import END, StateGraph

from nodes import act_node, reason_node
from react_state import AgentState

REASON = "reason"
ACT = "act"


def should_continue(state: AgentState):
    """Stop after a final answer; otherwise run the requested tools."""
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    return ACT


def build_graph():
    """Create the reason -> act -> reason loop."""
    graph = StateGraph(AgentState)

    graph.add_node(REASON, reason_node)
    graph.add_node(ACT, act_node)

    graph.set_entry_point(REASON)
    graph.add_conditional_edges(REASON, should_continue)
    graph.add_edge(ACT, REASON)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    result = app.invoke(
        {
            "input": "How many days ago was the latest SpaceX launch?",
            "agent_outcome": None,
            "intermediate_steps": [],
        },
        # Stop if the model keeps calling tools forever.
        config={"recursion_limit": 25},
    )

    print(result["agent_outcome"].return_values["output"])
