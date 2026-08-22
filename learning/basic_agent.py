from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    question: str
    answer: str


llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


def think(state: AgentState):

    response = llm.invoke(state["question"])

    return {
        "answer": response.content
    }


builder = StateGraph(AgentState)

builder.add_node("think", think)

builder.set_entry_point("think")

builder.add_edge("think", END)

agent = builder.compile()