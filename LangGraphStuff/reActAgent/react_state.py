"""The state that is passed between the graph nodes."""

import operator
from typing import Annotated, TypedDict

from langchain_core.agents import AgentAction, AgentFinish


class AgentState(TypedDict):
    """Data shared by the reason and act nodes."""

    input: str

    # Either a tool call or the final answer.
    agent_outcome: AgentAction | AgentFinish | None

    # A list of (tool call, tool result) pairs.
    # operator.add means new steps are added to the list.
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
