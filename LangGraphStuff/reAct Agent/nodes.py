import json

from dotenv import load_dotenv

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_reason_runnable import react_agent_runnable, tools
from react_state import AgentState

load_dotenv()


def _message_text(message: AIMessage) -> str:
    """Extract plain text from an AI message, including block-based content."""
    text = getattr(message, "text", "")
    if callable(text):
        text = text()
    if isinstance(text, str) and text:
        return text

    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in ("text", "output_text"):
                parts.append(block.get("text", ""))
        return "".join(parts)
    return ""


def _tool_args(tool_name: str, tool_input) -> dict:
    """Normalize a tool input into the dict shape required by tool calls."""
    if isinstance(tool_input, dict):
        return tool_input

    tool = next((t for t in tools if t.name == tool_name), None)
    schema = getattr(tool, "args", None) or {}
    if len(schema) == 1:
        return {next(iter(schema)): tool_input}
    return {"input": tool_input}


def _stringify(output) -> str:
    """Convert tool output into readable text for a ToolMessage."""
    if isinstance(output, str):
        return output
    try:
        return json.dumps(output, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(output)


def reason_node(state: AgentState):
    # Build messages from state
    messages = [HumanMessage(content=state["input"])]

    # Reconstruct conversation history from intermediate steps
    for i, (agent_action, observation) in enumerate(state.get("intermediate_steps", [])):
        tool_call_id = f"call_{i}"
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": tool_call_id,
                        "name": agent_action.tool,
                        "args": _tool_args(agent_action.tool, agent_action.tool_input),
                    }
                ],
            )
        )
        messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))

    response = react_agent_runnable.invoke(messages)
    output = _message_text(response)

    # Retry once if the model returns an empty response
    if not output and not response.tool_calls:
        response = react_agent_runnable.invoke(messages)
        output = _message_text(response)

    if response.tool_calls:
        # Keep every requested tool call instead of silently dropping extras.
        actions = [
            AgentAction(
                tool=tool_call["name"],
                tool_input=_tool_args(tool_call["name"], tool_call["args"]),
                log=f"Calling tool: {tool_call['name']}",
            )
            for tool_call in response.tool_calls
        ]
        return {"agent_outcome": actions}

    output = output or "The model returned an empty response."
    return {
        "agent_outcome": AgentFinish(
            return_values={"output": output},
            log=output,
        )
    }


def act_node(state: AgentState):
    outcome = state.get("agent_outcome")

    if isinstance(outcome, AgentAction):
        actions = [outcome]
    elif isinstance(outcome, list) and all(
        isinstance(action, AgentAction) for action in outcome
    ):
        actions = outcome
    else:
        raise ValueError(f"act_node expected an AgentAction, got: {outcome!r}")

    steps = []
    for agent_action in actions:
        tool_function = next(
            (t for t in tools if t.name == agent_action.tool), None
        )

        if tool_function is None:
            output = f"Tool '{agent_action.tool}' not found."
        else:
            try:
                output = tool_function.invoke(
                    _tool_args(agent_action.tool, agent_action.tool_input)
                )
            except Exception as exc:
                # Surface tool failures to the model so it can recover.
                output = f"Error running tool '{agent_action.tool}': {exc}"

        steps.append((agent_action, _stringify(output)))

    return {"intermediate_steps": steps}
