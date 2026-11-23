"""The two nodes of the ReAct agent: reason and act."""

from dotenv import load_dotenv

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_reason_runnable import llm_with_tools, tools
from react_state import AgentState

load_dotenv()


def reason_node(state: AgentState):
    """Ask the model what to do next."""
    print("Thinking...")

    # Start with the user's question.
    messages = [HumanMessage(content=state["input"])]

    # Add previous tool calls and results so the model remembers them.
    for i, (action, observation) in enumerate(state.get("intermediate_steps", [])):
        call_id = f"call_{i}"

        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": call_id,
                        "name": action.tool,
                        "args": action.tool_input,
                    }
                ],
            )
        )
        messages.append(ToolMessage(content=observation, tool_call_id=call_id))

    response = llm_with_tools.invoke(messages)

    # If the model asked for a tool, send that request to act_node.
    if response.tool_calls:
        # For simplicity, run the first requested tool.
        call = response.tool_calls[0]
        return {
            "agent_outcome": AgentAction(
                tool=call["name"],
                tool_input=call["args"],
                log=f"Calling {call['name']}",
            )
        }

    # Otherwise the model has given its final answer.
    answer = response.text or "Sorry, I could not find an answer."
    return {
        "agent_outcome": AgentFinish(
            return_values={"output": answer},
            log=answer,
        )
    }


def act_node(state: AgentState):
    """Run the requested tool."""
    action = state["agent_outcome"]
    print(f"Using tool: {action.tool}")

    # Find the tool by name.
    tool_to_use = next((t for t in tools if t.name == action.tool), None)

    if tool_to_use is None:
        result = f"Tool '{action.tool}' was not found."
    else:
        try:
            result = tool_to_use.invoke(action.tool_input)
        except Exception as error:
            result = f"Tool error: {error}"

    return {"intermediate_steps": [(action, str(result))]}
