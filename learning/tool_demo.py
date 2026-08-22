from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage

from tools.calculator import calculator

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

llm_with_tools = llm.bind_tools([calculator])

messages = [
    HumanMessage(
        content="What is 145 + 278?"
    )
]

# First LLM call: decide whether a tool is needed.
ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

print("AI requested:", ai_message.tool_calls)

# Execute every requested tool and send its result back to the model.
for tool_call in ai_message.tool_calls:
    result = calculator.invoke(tool_call["args"])

    messages.append(
        ToolMessage(
            content=str(result),
            tool_call_id=tool_call["id"],
        )
    )

if not ai_message.tool_calls:
    print(ai_message.content)
else:
    # Second LLM call: produce a user-facing answer from the tool result.
    final_response = llm_with_tools.invoke(messages)

    print(final_response.content)
