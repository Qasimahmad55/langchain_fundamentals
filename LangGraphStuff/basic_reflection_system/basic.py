
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain

graph = MessageGraph()


def generate(state):
    print("\n⚡ Generating tweet...")
    return generation_chain.invoke({"messages": state})


def reflect(state):
    print("\n🧐 Reviewing tweet...")

    tweet = state[-1].content

    response = reflection_chain.invoke({
        "messages": [
            HumanMessage(content=f"Critique this tweet and suggest improvements:\n{tweet}")
        ]
    })

    return HumanMessage(content=response.content)


def should_continue(state):
    return END if len(state) > 2 else "reflect"


graph.add_node("generate", generate)
graph.add_node("reflect", reflect)

graph.set_entry_point("generate")
graph.add_conditional_edges("generate", should_continue)
graph.add_edge("reflect", "generate")

app = graph.compile()


if __name__ == "__main__":
    # print(app.get_graph().draw_mermaid())

    response = app.invoke([
        HumanMessage(content="AI agents taking over content creation")
    ])

    print("\n========== FINAL RESULT ==========")
    print(response[-1].content[0]['text'])
