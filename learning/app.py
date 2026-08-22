from langchain_core.messages import HumanMessage

from learning.langgraph_agent import graph

response = graph.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is 145 + 278?"
            )
        ]
    }
)

for message in response["messages"]:
    print(message)