from dotenv import load_dotenv

from agent_reason_runnable import react_agent_runnable, tools
from react_state import AgentState

load_dotenv()


def reason_node(state: AgentState):
    """Ask the agent runnable for the next action or final answer."""
    agent_outcome = react_agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}


def act_node(state: AgentState):
    """Run the single tool chosen in agent_outcome and record the output."""
    agent_action = state["agent_outcome"]

    # Extract tool name and input from AgentAction
    tool_name = agent_action.tool
    tool_input = agent_action.tool_input

    # Find the matching tool
    tool_function = None
    for tool in tools:
        if tool.name == tool_name:
            tool_function = tool
            break

    # Run the tool if found
    if tool_function is not None:
        try:
            if isinstance(tool_input, dict):
                output = tool_function.invoke(tool_input)
            else:
                output = tool_function.invoke(tool_input)
        except Exception as err:
            output = f"Error running tool '{tool_name}': {err}"
    else:
        output = f"Tool '{tool_name}' not found"

    return {"intermediate_steps": [(agent_action, str(output))]}
