import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

import sqlite3
from typing import Annotated, Iterator, TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from config import OLLAMA_MODEL, DB_PATH
from eval_hook import RunRecorder
from tools.football import (
    football_countries,
    find_team,
    team_squad,
    player_details,
    team_details,
    league_table,
    top_scorers,
)
from tools.knowledge import search_knowledge

tools = [
    football_countries,
    find_team,
    team_squad,
    player_details,
    team_details,
    league_table,
    top_scorers,
    search_knowledge,
]

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a football data assistant. Answer using tool results only; if a tool "
    "returns an error or no data, say the lookup failed — never invent players, "
    "numbers, standings, or IDs. To look up a team, ALWAYS call find_team first to "
    "get its real id, then pass that id to other tools. Never guess a team_id. "
    "For questions about rules, competition formats, or football background (not "
    "live data), call search_knowledge and answer only from what it returns, citing "
    "the source file in square brackets. "
    "For messages that aren't data requests (e.g. a user saying which team they "
    "support), reply briefly and ask what they'd like to know."
))

# What the agent is told about itself, sent with each run so the evaluator can check self-descriptions.
INSTRUCTIONS = SYSTEM_PROMPT.content + "\n\nTools:\n" + "\n".join(
    f"- {t.name}: {' '.join(t.description.split())}" for t in tools
)

MODEL = OLLAMA_MODEL
llm = ChatOllama(model=MODEL, temperature=0)
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def think(state: AgentState) -> dict:
    response = llm_with_tools.invoke([SYSTEM_PROMPT] + state["messages"])
    return {"messages": [response]}

# -------------------------------------------------------------------
# Graph Builder
# -------------------------------------------------------------------
builder = StateGraph(AgentState)

# Nodes
builder.add_node("think", think)
builder.add_node("tools", ToolNode(tools)) 

# Edges ( connections between nodes )
builder.add_edge(START, "think")
builder.add_conditional_edges("think", tools_condition)  # If tool requested → tools, else END
builder.add_edge("tools", "think")  # After tool runs, return to think

# SQLite checkpointer — state is written to a file, so memory survives a restart.
# check_same_thread=False because LangGraph may touch the connection from another thread.
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
memory = SqliteSaver(conn)
agent = builder.compile(checkpointer=memory)


def run_events(text: str, thread_id: str) -> Iterator[dict]:
    """Stream one turn as structured events so any front-end (CLI, HTTP) can render it.

    Yields dicts: {"type": "tool_call", "names": [...]},
                  {"type": "tool_result", "name": str},
                  {"type": "token", "content": str}.
    """
    config = {"configurable": {"thread_id": thread_id}}
    recorder = RunRecorder(text, thread_id, MODEL, INSTRUCTIONS)

    # "updates" gives node-level tool markers; "messages" gives per-token output.
    for mode, data in agent.stream(
        {"messages": [HumanMessage(content=text)]},
        config=config,
        stream_mode=["updates", "messages"],
    ):
        if mode == "updates":
            recorder.update(data)
            for node, update in data.items():
                message = update["messages"][-1]
                if node == "think" and getattr(message, "tool_calls", None):
                    yield {"type": "tool_call", "names": [c["name"] for c in message.tool_calls]}
                elif node == "tools":
                    yield {"type": "tool_result", "name": message.name}
        else:  # "messages" — one token chunk at a time
            token, meta = data
            if meta.get("langgraph_node") == "think" and token.content:
                yield {"type": "token", "content": token.content}

    # The turn is complete: send it for evaluation (only if EVAL_API_URL is set).
    recorder.finish()


# -------------------------------------------------------------------
# Main Execution — interactive CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # thread_id picks which stored conversation to resume. Pass a new one on the
    # command line for a clean slate that forces fresh tool calls.
    thread_id = sys.argv[1] if len(sys.argv) > 1 else "joe"

    print(f"Football assistant (model: {MODEL}, thread: {thread_id}). Type 'exit' or Ctrl-C to quit.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break

        started = False
        for event in run_events(text, thread_id):
            if event["type"] == "tool_call":
                print("  …tool call:", event["names"])
            elif event["type"] == "tool_result":
                print("  …tool returned:", event["name"])
            elif event["type"] == "token":
                if not started:
                    print("agent> ", end="", flush=True)
                    started = True
                print(event["content"], end="", flush=True)

        print("\n")
