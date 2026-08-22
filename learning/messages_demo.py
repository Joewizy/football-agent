from langchain_ollama import ChatOllama
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

messages = [
    SystemMessage(
        content="You are a financial advisor."
    ),
    HumanMessage(
        content="What is the price of one dollar in naira today?"
    ),
    AIMessage(
        content="The price of one dollar in naira today is 100 naira."
    ),
    HumanMessage(
        content="I have 450 usd, how much is that in naira?"
    ),
]

response = llm.invoke(messages)

print(response.content)