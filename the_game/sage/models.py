import json
import os
import time
import requests
from dotenv import load_dotenv
from .config import BASE_DIR, MODELS_DIR, MAX_RETRIES, MODEL_TIMEOUT

load_dotenv(BASE_DIR / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_all_models():
    """Load and merge all model JSON files, sorted by model_id."""
    models = []
    for f in sorted(MODELS_DIR.glob("*.json")):
        with open(f) as fh:
            models.extend(json.load(fh))
    return sorted(models, key=lambda m: m["model_id"])


def get_model_info(model_id, models):
    """Find a model dict by its model_id."""
    for m in models:
        if m["model_id"] == model_id:
            return m
    return None


# Event set when a timeout fires — signals the generate thread to stop retrying
import threading
_cancel_flags = threading.local()


def generate(model_id, provider, messages, temperature=0.7, base_url=None):
    """Unified generate with retry + exponential backoff."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        # Check if this call has been cancelled by timeout
        if getattr(_cancel_flags, 'cancelled', False):
            raise TimeoutError(f"{model_id} cancelled")
        try:
            if provider == "openrouter":
                return _openrouter(model_id, messages, temperature)
            else:
                return _ollama(model_id, messages, temperature, base_url or "http://localhost:11434")
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
    raise last_err


def _openrouter(model_id, messages, temperature):
    resp = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        },
        timeout=MODEL_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _ollama(model_id, messages, temperature, base_url):
    resp = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model_id,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": 2048},
        },
        timeout=MODEL_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]
