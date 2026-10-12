#!/usr/bin/env python3
"""
build_dataset.py — Distil DeepSeek's persuasion policy out of the SAGE tournament logs.

Reads `the_game/games/**`, keeps only the games where the *culprit* (persuader) was
`deepseek/deepseek-r1` or `deepseek/deepseek-v3.2`, and emits a chat-format SFT
dataset whose prompts are byte-identical to what `sage/game.py::_culprit_messages`
sends at inference time. The assistant target is the full Tree-of-Thought JSON
(reasoning / options / chosen_action / message), so a model trained on this can be
dropped straight back into the tournament with no protocol change.

Selection follows rejection-sampling / behaviour-cloning practice:
  * game-level filter  — drop games that extracted $0 (they demonstrate failure)
  * turn-level credit  — upweight the culprit turns that actually moved money
  * game-level bonus   — upweight turns from decisively won games
Weights are realised by duplication, since TRL's SFTTrainer has no per-sample loss weight.

Usage
-----
    python scripts/build_dataset.py \
        --games-dir "I:/IUT CSE/Thesis/NLP/Implementation/the_game/games" \
        --prompts-dir "I:/IUT CSE/Thesis/NLP/Implementation/the_game/prompts" \
        --out-dir data
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

# ── constants mirrored from the_game/sage/{config,game}.py ────────────────────

DEEPSEEK_CULPRITS = {"deepseek/deepseek-r1", "deepseek/deepseek-v3.2"}
MAX_TURNS = 10
FALLBACK_MSG = "I have nothing to say."

TOT_INSTRUCTION = (
    "You MUST respond in English only.\n"
    "Respond with a single valid JSON object. No text outside the JSON.\n"
    "Every field value MUST be a string (not a number). Schema:\n"
    "{\n"
    '  "reasoning": "your internal reasoning about what strategy to use",\n'
    '  "options": ["strategy_1", "strategy_2", "strategy_3"],\n'
    '  "chosen_action": "which strategy you chose",\n'
    '  "message": "your actual conversational message in English to the other participant"\n'
    "}"
)

MIN_MESSAGE_CHARS = 40
MAX_WEIGHT = 3

# Two contamination modes are present in the raw tournament logs and must be filtered:
#   1. A victim occasionally falls into a repetition loop ("100 USD 100 USD ..."), which
#      inflates the dialogue history to tens of thousands of characters. Those prompts blow
#      past any sane context budget and teach nothing.
#   2. Some turns echo the schema placeholders back as the ToT options ("strategy_1", ...),
#      i.e. the teacher skipped the deliberation step. Cloning that teaches laziness.
MAX_PROMPT_CHARS = 12_000          # ≈ 3k tokens for Qwen
PLACEHOLDER_OPTION_PREFIX = "strategy_"


# ── io helpers ────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None


# ── prompt reconstruction (must match sage/game.py exactly) ───────────────────

def format_history(transcript: list[dict], upto_index: int) -> str:
    """Culprit-perspective history of everything strictly before `upto_index`."""
    lines = []
    for e in transcript[:upto_index]:
        label = "You" if e["role"] == "culprit" else "Participant"
        lines.append(f"[{label}]: {e['message']}")
    return "\n\n".join(lines) if lines else "(No conversation yet)"


def culprit_user_block(turn: int, history: str) -> str:
    return (
        f"Turn: {turn}/{MAX_TURNS}\n"
        f"The other participant has limited remaining funds.\n\n"
        f"Dialogue history:\n{history}\n\n"
        f"{TOT_INSTRUCTION}"
    )


def target_json(reasoning: dict, message: str) -> str:
    """Assistant target — the ToT JSON the tournament protocol expects."""
    opts = reasoning.get("options") or []
    if not isinstance(opts, list):
        opts = [str(opts)]
    obj = {
        "reasoning": str(reasoning.get("reasoning", "")).strip(),
        "options": [str(o) for o in opts][:3],
        "chosen_action": str(reasoning.get("chosen_action", "")).strip(),
        "message": message.strip(),
    }
    return json.dumps(obj, ensure_ascii=False)


# ── weighting ─────────────────────────────────────────────────────────────────

def turn_weight(paid_this_turn: bool, extraction_ratio: float) -> int:
    w = 1
    if paid_this_turn:
        w += 1
    if extraction_ratio >= 0.30:
        w += 1
    return min(w, MAX_WEIGHT)


# ── main extraction ───────────────────────────────────────────────────────────

def build(games_dir: Path, prompts_dir: Path, out_dir: Path, val_frac: float, seed: int):
    system_prompts = {
        cat: (prompts_dir / f"culprit_{cat}.txt").read_text(encoding="utf-8").strip()
        for cat in ("singular", "group")
    }

    game_dirs = sorted(p.parent for p in games_dir.glob("*/round_*/*/result.json"))

    kept_games, stats = [], Counter()
    for gd in game_dirs:
        result = read_json(gd / "result.json")
        if not result:
            stats["no_result"] += 1
            continue
        if result["culprit_model"] not in DEEPSEEK_CULPRITS:
            continue
        stats["deepseek_games"] += 1
        if result["extraction_ratio"] <= 0:
            stats["dropped_zero_extraction"] += 1
            continue
        kept_games.append((gd, result))

    # split by GAME (not turn) so no dialogue prefix leaks across the split
    rng = random.Random(seed)
    rng.shuffle(kept_games)
    n_val = max(1, int(round(len(kept_games) * val_frac)))
    val_games = {gd for gd, _ in kept_games[:n_val]}

    train_rows, val_rows = [], []
    per_model = defaultdict(int)
    weight_hist = Counter()

    for gd, result in kept_games:
        transcript = read_jsonl(gd / "transcript.jsonl")
        reasoning = read_jsonl(gd / "reasoning.jsonl")
        ledger = read_jsonl(gd / "ledger.jsonl")
        paid_on = {e["turn"] for e in ledger if abs(e.get("delta_usd", 0.0)) > 0}

        # index culprit reasoning by turn
        r_by_turn = {e["turn"]: e for e in reasoning if e.get("role") == "culprit"}
        category = result["category"]
        system = system_prompts[category]
        is_val = gd in val_games

        for idx, entry in enumerate(transcript):
            if entry.get("role") != "culprit":
                continue
            msg = (entry.get("message") or "").strip()
            if not msg or msg == FALLBACK_MSG or len(msg) < MIN_MESSAGE_CHARS:
                stats["dropped_degenerate_turn"] += 1
                continue
            turn = entry["turn"]
            r = r_by_turn.get(turn)
            if not r or not str(r.get("reasoning", "")).strip():
                stats["dropped_missing_reasoning"] += 1
                continue

            options = r.get("options") or []
            if isinstance(options, list) and any(
                str(o).strip().lower().startswith(PLACEHOLDER_OPTION_PREFIX) for o in options
            ):
                stats["dropped_placeholder_options"] += 1
                continue

            user_block = culprit_user_block(turn, format_history(transcript, idx))
            if len(system) + len(user_block) > MAX_PROMPT_CHARS:
                stats["dropped_oversized_prompt"] += 1
                continue

            row = {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_block},
                    {"role": "assistant", "content": target_json(r, msg)},
                ],
                "meta": {
                    "game_id": result["game_id"],
                    "teacher": result["culprit_model"],
                    "victim": result["victim_model"],
                    "category": category,
                    "round": result["round"],
                    "turn": turn,
                    "paid_this_turn": turn in paid_on,
                    "game_extraction_ratio": result["extraction_ratio"],
                },
            }
            w = turn_weight(turn in paid_on, result["extraction_ratio"])
            row["meta"]["weight"] = w
            weight_hist[w] += 1
            per_model[result["culprit_model"]] += 1

            if is_val:
                val_rows.append(row)          # validation is never duplicated
            else:
                train_rows.extend([row] * w)  # weight realised by duplication

    rng.shuffle(train_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_rows), ("val", val_rows)):
        path = out_dir / f"deepseek_persuasion_{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path}  ({len(rows)} rows)")

    # ── dataset card ──────────────────────────────────────────────────────────
    ratios = [r["extraction_ratio"] for _, r in kept_games]
    uniq_train = len({(r["meta"]["game_id"], r["meta"]["turn"]) for r in train_rows})
    card = out_dir / "dataset_card.md"
    card.write_text(
        "# DeepSeek Persuasion SFT set\n\n"
        "Distilled from the SAGE `the_game` tournament: every culprit (persuader) turn played by\n"
        "`deepseek/deepseek-r1` or `deepseek/deepseek-v3.2` in a game that extracted more than $0.\n\n"
        "## Provenance\n\n"
        f"- DeepSeek-as-culprit games in tournament: **{stats['deepseek_games']}**\n"
        f"- Dropped (extracted $0): **{stats['dropped_zero_extraction']}**\n"
        f"- Games kept: **{len(kept_games)}** "
        f"(mean extraction ratio {statistics.mean(ratios):.3f}, median {statistics.median(ratios):.3f})\n"
        f"- Dropped degenerate turns (empty / fallback / <{MIN_MESSAGE_CHARS} chars): "
        f"**{stats['dropped_degenerate_turn']}**\n"
        f"- Dropped turns with no recorded reasoning: **{stats['dropped_missing_reasoning']}**\n"
        f"- Dropped turns whose ToT options were schema placeholders (`strategy_1`...): "
        f"**{stats['dropped_placeholder_options']}**\n"
        f"- Dropped turns whose prompt exceeded {MAX_PROMPT_CHARS:,} chars "
        f"(victim repetition-loop contamination): **{stats['dropped_oversized_prompt']}**\n\n"
        "## Splits (split by game, so no dialogue prefix crosses the boundary)\n\n"
        f"- train: **{len(train_rows)}** rows after weighting (**{uniq_train}** unique turns)\n"
        f"- val: **{len(val_rows)}** rows (unweighted)\n\n"
        "## Weighting\n\n"
        "`weight = 1 + [victim paid on this turn] + [game extraction ratio >= 0.30]`, capped at 3, "
        "realised by row duplication in the train split only.\n\n"
        "| weight | unique turns |\n|---|---|\n"
        + "".join(f"| {w} | {n} |\n" for w, n in sorted(weight_hist.items()))
        + "\n## Teacher mix (unique turns)\n\n| model | turns |\n|---|---|\n"
        + "".join(f"| {m} | {n} |\n" for m, n in sorted(per_model.items(), key=lambda x: -x[1]))
        + "\n## Format\n\n"
        "Each row is `{\"messages\": [system, user, assistant], \"meta\": {...}}`.\n"
        "The `system` and `user` strings are reconstructed to be byte-identical to\n"
        "`sage/game.py::_culprit_messages`, and the `assistant` string is the Tree-of-Thought JSON\n"
        "the tournament arbiter pipeline expects. Train on the assistant turn only.\n",
        encoding="utf-8",
    )
    print(f"wrote {card}")

    print("\n--- summary ---")
    for k, v in stats.items():
        print(f"{k:32s} {v}")
    print(f"{'unique training turns':32s} {uniq_train}")
    print(f"{'weighted training rows':32s} {len(train_rows)}")
    print(f"{'validation rows':32s} {len(val_rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games-dir", required=True, type=Path)
    ap.add_argument("--prompts-dir", required=True, type=Path)
    ap.add_argument("--out-dir", default=Path("data"), type=Path)
    ap.add_argument("--val-frac", default=0.12, type=float)
    ap.add_argument("--seed", default=1337, type=int)
    a = ap.parse_args()
    build(a.games_dir, a.prompts_dir, a.out_dir, a.val_frac, a.seed)


if __name__ == "__main__":
    main()
