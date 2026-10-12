import json
import re
from .config import ARBITER_MODEL, ARBITER_PROVIDER, TEMP_AR1, TEMP_AR2, TEMP_AR3, PROMPTS_DIR
from .models import generate

_arbiter_prompt = None


def _get_prompt():
    global _arbiter_prompt
    if _arbiter_prompt is None:
        _arbiter_prompt = (PROMPTS_DIR / "arbiter.txt").read_text()
    return _arbiter_prompt


def run_consensus(victim_message, victim_turn):
    """Run 2-arbiter consensus; invoke tie-breaker AR3 if they disagree."""
    messages = [
        {"role": "system", "content": _get_prompt()},
        {"role": "user", "content": f'Victim message (Turn {victim_turn}):\n"{victim_message}"'},
    ]

    r1 = _call_arbiter(messages, TEMP_AR1)
    r2 = _call_arbiter(messages, TEMP_AR2)

    v1 = _parse(r1, victim_turn)
    v2 = _parse(r2, victim_turn)

    ar3_result = None
    if v1["given_usd"] == v2["given_usd"]:
        final = v1
        consensus = True
    else:
        r3 = _call_arbiter(messages, TEMP_AR3)
        ar3_result = _parse(r3, victim_turn)
        final = ar3_result
        consensus = False

    return {
        "ar1": v1,
        "ar2": v2,
        "ar3": ar3_result,
        "final": final,
        "consensus": consensus,
    }


def _call_arbiter(messages, temperature):
    try:
        return generate(ARBITER_MODEL, ARBITER_PROVIDER, messages, temperature)
    except Exception:
        return '{"given_usd": 0, "confidence": 0, "evidence_quote": ""}'


def _parse(response, victim_turn):
    default = {
        "victim_turn": victim_turn,
        "given_usd": 0.0,
        "transaction_type": "none",
        "confidence": 0.0,
        "evidence_quote": "",
    }
    try:
        match = re.search(r"\{[^{}]+\}", response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            for k in default:
                if k in data:
                    default[k] = data[k]
            if default["given_usd"] < 0:
                default["given_usd"] = 0.0
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return default
