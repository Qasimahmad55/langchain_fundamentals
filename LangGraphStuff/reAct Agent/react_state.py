"""The state that is passed between the graph nodes."""

import operator
from typing import Annotated, TypedDict

from langchain_core.agents import AgentAction, AgentFinish


class AgentState(TypedDict):
    """Data shared by the reason and act nodes."""

    input: str

    # Either the tools the model wants to call, or its final answer.
    agent_outcome: list[AgentAction] | AgentFinish | None

    # A list of (tool call, tool result) pairs. operator.add appends new steps.
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
