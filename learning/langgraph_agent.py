from typing import Annotated, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from tools.calculator import calculator

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

llm = llm.bind_tools([calculator])


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State):
    response = llm.invoke(state["messages"])

    # The LLM's raw decision — which tool, which args — before anything runs.
    print("\n--- LLM tool_calls ---")
    print(response.tool_calls)
    print("----------------------\n")

    return {"messages": [response]}


builder = StateGraph(State)

builder.add_node("chatbot", chatbot)
builder.add_node("tools", ToolNode([calculator]))

builder.set_entry_point("chatbot")

builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

builder.add_edge("tools", "chatbot")

graph = builder.compile()


if __name__ == "__main__":
    result = graph.invoke(
        {"messages": [HumanMessage(content="What is 12 plus 30?")]}
    )
    print("Final answer:", result["messages"][-1].content)