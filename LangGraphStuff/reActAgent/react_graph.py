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
    """Stop after the final answer; otherwise run the tool."""
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


def main():
    """Run the agent with one question."""
    app = build_graph()

    result = app.invoke(
        {
            "input": "How many days ago was the latest SpaceX launch?",
            "agent_outcome": None,
            "intermediate_steps": [],
        },
        # Stop if the model keeps calling tools forever.
        config={"recursion_limit": 15},
    )

    print("\nFinal answer:")
    print(result["agent_outcome"].return_values["output"])


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nThe agent could not finish.")
        print("Error:", error)

    # Keep the window open when the file is double-clicked.
    input("\nPress Enter to close...")
