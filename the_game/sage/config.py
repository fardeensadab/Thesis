from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
PROMPTS_DIR = BASE_DIR / "prompts"
GAMES_DIR = BASE_DIR / "games"
ELO_DIR = BASE_DIR / "elo"
LOGS_DIR = BASE_DIR / "logs"

# Arbiter — always grok 4.1 fast via OpenRouter
ARBITER_MODEL = "x-ai/grok-4.1-fast"
ARBITER_PROVIDER = "openrouter"

# Temperature
TEMP_CULPRIT = 0.7
TEMP_VICTIM = 0.7
TEMP_AR1 = 0.2
TEMP_AR2 = 0.2
TEMP_AR3 = 0.0

# Game
INITIAL_USD = 100.0
MAX_TURNS = 10

# Elo
ELO_K = 24
ELO_DEFAULT = 1500.0

# API
MAX_RETRIES = 3
MODEL_TIMEOUT = 120  # seconds — terminate game if a model call exceeds this

# Tournament
CATEGORIES = ["singular", "group"]
ROUNDS_PER_CATEGORY = 5

# Fallback message on API failure
FALLBACK_MSG = "I have nothing to say."

# Ensure directories exist
for _d in [GAMES_DIR, ELO_DIR, LOGS_DIR, PROMPTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
