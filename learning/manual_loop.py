from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, ToolMessage

from tools.football import (
    football_countries,
    find_team,
    team_squad,
    player_details,
)


tools = [football_countries, find_team, team_squad, player_details]
tools_by_name = {tool.name: tool for tool in tools}

llm = ChatOllama(
    model="llama3.2",
    temperature=0,
)

# Tell the model the tools exist (this runs nothing).
llm_with_tools = llm.bind_tools(tools)


messages = [
    HumanMessage(content="What football countries are available?")
]

# First call: the LLM may reply by asking for a tool instead of answering.
ai_response = llm_with_tools.invoke(messages)
messages.append(ai_response)
print("AI response: ", ai_response)


# tool_calls is non-empty when the LLM wants a tool run.
if ai_response.tool_calls:

    print("The LLM requested one or more tools:")

    for tool_call in ai_response.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"  -> running tool '{tool_name}' with arguments {tool_args}")

        selected_tool = tools_by_name[tool_name]

        # This is where the tool (and the real SportMonks call) actually runs.
        tool_result = selected_tool.invoke(tool_args)

        # Hand the result back, tagged so the LLM knows which request it answers.
        messages.append(
            ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            )
        )

    # Second call: the LLM can now see the tool output and write the answer.
    final_response = llm_with_tools.invoke(messages)

    print("\nFinal answer from the LLM:")
    print(final_response.content)

else:
    print("The LLM answered directly (no tool was used):")
    print(ai_response.content)
