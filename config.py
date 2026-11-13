"""Single source of runtime configuration. Read the environment once, here."""
import os

from dotenv import load_dotenv

load_dotenv()

_ROOT = os.path.dirname(os.path.abspath(__file__))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")

# Checkpointer file. Overridable so tests/deploys can point elsewhere.
DB_PATH = os.getenv("FOOTBALL_DB_PATH", os.path.join(_ROOT, "football_memory.sqlite"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Optional: an agent evaluation API that scores every turn (see eval_hook.py). Unset = off.
EVAL_API_URL = os.getenv("EVAL_API_URL")
# `or`, not a getenv default: an empty "EVAL_AGENT_NAME=" line in .env would otherwise send a blank name.
EVAL_AGENT_NAME = os.getenv("EVAL_AGENT_NAME") or "football-agent"
