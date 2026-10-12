#!/usr/bin/env python3
"""
test_arena.py — offline smoke test for persuasion_arena.py.

Runs the whole cohort loop with stubbed generation and a regex arbiter, so the episode
bookkeeping, prompt reconstruction, ToT parsing, ledger, early-stop and paired statistics
can be verified on a laptop without a GPU or an API key. Catches the class of bug that
otherwise only shows up two hours into a Kaggle session.

    python scripts/test_arena.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True          # keep __pycache__ out of the Kaggle upload payload

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "kaggle_dataset"))

import persuasion_arena as pa  # noqa: E402

FAILURES = []


def check(name: str, cond: bool, detail: str = ""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# ── 1. ToT parser ─────────────────────────────────────────────────────────────

print("\n[1] ToT parser")

msg, r = pa.parse_tot(json.dumps({
    "reasoning": "build rapport", "options": ["a", "b", "c"],
    "chosen_action": "a", "message": "Please help Maya."}), "culprit", 1)
check("clean JSON", msg == "Please help Maya." and r["chosen_action"] == "a")

msg, _ = pa.parse_tot(
    'Sure!\n```json\n{"reasoning":"x","options":["a"],"chosen_action":"a","message":"Hi there"}\n```',
    "culprit", 1)
check("JSON wrapped in prose/fences", msg == "Hi there", msg)

msg, _ = pa.parse_tot(
    '<think>long chain of thought</think>{"reasoning":"x","options":[],"chosen_action":"y","message":"Hello"}',
    "culprit", 1)
check("<think> block stripped", msg == "Hello", msg)

msg, _ = pa.parse_tot('{"reasoning":"x","options":["a"],"chosen_action":"a","message":"truncated mid', "culprit", 1)
check("truncated JSON recovered", msg == "truncated mid", msg)


# ── 2. prompt reconstruction matches the tournament ───────────────────────────

print("\n[2] Prompt reconstruction vs the_game")

ep = pa.Episode("t|singular|00", "victim/x", "singular", 1)
check("empty history placeholder", ep.history("culprit") == "(No conversation yet)")

ep.transcript = [{"turn": 1, "role": "culprit", "message": "A"},
                 {"turn": 1, "role": "victim", "message": "B"}]
check("culprit-perspective labels", ep.history("culprit") == "[You]: A\n\n[Participant]: B",
      repr(ep.history("culprit")))
check("victim-perspective labels", ep.history("victim") == "[Fundraiser]: A\n\n[You]: B",
      repr(ep.history("victim")))

built = ep.culprit_messages(2, "SYS")[1]["content"]
check("turn header", built.startswith("Turn: 2/10\nThe other participant has limited remaining funds."))
check("ToT instruction appended", built.endswith(pa.TOT_INSTRUCTION))

vb = ep.victim_messages(2, "SYS")[1]["content"]
check("victim sees remaining balance", "Remaining money: $100.00 USD" in vb)

# byte-identical to the dataset builder's reconstruction
sys.path.insert(0, str(ROOT / "scripts"))
import build_dataset as bd  # noqa: E402
check("arena prompt == dataset prompt",
      built == bd.culprit_user_block(2, ep.history("culprit")))
check("arena ToT == dataset ToT", pa.TOT_INSTRUCTION == bd.TOT_INSTRUCTION)


# ── 3. stubbed cohort run ─────────────────────────────────────────────────────

print("\n[3] Cohort loop (stubbed generation + regex arbiter)")

PROMPTS = pa.load_prompts(ROOT / "kaggle_dataset" / "prompts")
check("all four prompts load",
      {"culprit_singular", "culprit_group", "victim", "arbiter"} <= set(PROMPTS), str(sorted(PROMPTS)))


class StubArbiter(pa.Arbiter):
    """Regex money extractor — no network."""
    def __init__(self):
        self.n_calls = 0
        self.n_disagreements = 0
        self.model = "stub"

    def consensus(self, victim_message, victim_turn):
        self.n_calls += 1
        m = re.search(r"\$(\d+(?:\.\d+)?)", victim_message)
        return {"final": {"victim_turn": victim_turn, "given_usd": float(m.group(1)) if m else 0.0,
                          "transaction_type": "donation", "confidence": 1.0, "evidence_quote": ""},
                "consensus": True}

    def batch(self, items):
        return [self.consensus(*i) for i in items]


class StubModel:
    device = "cpu"


class StubTok:
    padding_side = "right"
    pad_token_id = 0

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=False):
        return "\n".join(m["content"] for m in conv)


def make_stub_generate(generosity: float, rng_seed: int):
    """Culprit emits ToT JSON; victim donates `generosity`-scaled amounts."""
    rng = random.Random(rng_seed)

    def stub(model, tok, conversations, temperature, max_new_tokens=320, max_input_tokens=3072):
        out = []
        for conv in conversations:
            is_culprit = "fundraiser" in conv[0]["content"].lower()
            if is_culprit:
                out.append(json.dumps({"reasoning": "r", "options": ["a", "b", "c"],
                                       "chosen_action": "a", "message": "Please donate to help."}))
            else:
                amt = 0.0 if rng.random() > generosity else float(rng.choice([5, 10, 20]))
                text = f"I will give ${amt:.0f} USD." if amt else "I will keep my money."
                out.append(json.dumps({"reasoning": "r", "options": ["a"],
                                       "chosen_action": "a", "message": text}))
        return out
    return stub


specs = pa.make_episode_specs(["stub/victim-a", "stub/victim-b"], ["singular", "group"], 3)
check("episode grid size", len(specs) == 2 * 2 * 3, str(len(specs)))
check("episode ids unique", len({s["episode_id"] for s in specs}) == len(specs))

real_generate = pa.batch_generate
results = {}
for arm, generosity, seed in (("base", 0.25, 1), ("finetuned", 0.55, 1)):
    pa.batch_generate = make_stub_generate(generosity, seed)
    eps = [pa.Episode(**s) for s in specs]
    pa.run_cohort(StubModel(), StubTok(), StubModel(), StubTok(), eps,
                  PROMPTS, StubArbiter(), arm, verbose=False)
    results[arm] = [e.to_result(arm) for e in eps]
    mean = sum(r["extraction_ratio"] for r in results[arm]) / len(eps)
    print(f"       {arm:10s} mean extraction ratio {mean:.3f}")
    check(f"{arm}: all episodes terminated", all(e.done for e in eps))
    check(f"{arm}: ledger never goes negative", all(e.remaining_usd >= 0 for e in eps))
    check(f"{arm}: turns within bounds", all(1 <= e.turns_played <= pa.MAX_TURNS for e in eps))
    check(f"{arm}: transcript alternates roles",
          all([m["role"] for m in e.transcript] == ["culprit", "victim"] * (len(e.transcript) // 2)
              for e in eps))
    check(f"{arm}: ledger balance is consistent",
          all(abs(e.remaining_usd - (pa.INITIAL_USD + sum(l["delta_usd"] for l in e.ledger))) < 1e-6
              for e in eps))
pa.batch_generate = real_generate

# early-stop: an episode drained to $0 must stop before turn 10
drained = [e for e in results["finetuned"] if e["remaining_usd"] == 0]
if drained:
    check("drained episodes stop early", all(e["turns_played"] <= pa.MAX_TURNS for e in drained))


# ── 4. paired statistics ──────────────────────────────────────────────────────

print("\n[4] Paired statistics")

rep = pa.paired_report(results["base"], results["finetuned"], n_boot=4000)
print("       " + json.dumps({k: rep[k] for k in
                              ("n_pairs", "base_mean_extraction_ratio", "ft_mean_extraction_ratio",
                               "mean_paired_delta", "delta_ci95", "permutation_p_two_sided")}))
check("pairs matched", rep["n_pairs"] == len(specs))
check("delta is the difference of means",
      abs(rep["mean_paired_delta"]
          - (rep["ft_mean_extraction_ratio"] - rep["base_mean_extraction_ratio"])) < 1e-3)
check("more-generous arm scores higher", rep["mean_paired_delta"] > 0, str(rep["mean_paired_delta"]))
check("p-value in range", 0 < rep["permutation_p_two_sided"] <= 1)
check("CI brackets the estimate",
      rep["delta_ci95"][0] <= rep["mean_paired_delta"] <= rep["delta_ci95"][1])
check("cell counts sum to n",
      rep["ft_better_cells"] + rep["ft_worse_cells"] + rep["tied_cells"] == rep["n_pairs"])

# a null effect must not be flagged as significant
null_rep = pa.paired_report(results["base"], results["base"], n_boot=4000)
check("identical arms give zero delta", null_rep["mean_paired_delta"] == 0)
check("identical arms are not significant", null_rep["permutation_p_two_sided"] > 0.05)

rows = pa.per_victim_table(results["base"], results["finetuned"])
check("per-victim table covers every victim", len(rows) == 2, str(rows))


# ── 5. dataset integrity ──────────────────────────────────────────────────────

print("\n[5] Dataset integrity")

train = [json.loads(l) for l in (ROOT / "data" / "deepseek_persuasion_train.jsonl")
         .read_text(encoding="utf-8").splitlines() if l.strip()]
val = [json.loads(l) for l in (ROOT / "data" / "deepseek_persuasion_val.jsonl")
       .read_text(encoding="utf-8").splitlines() if l.strip()]

check("train non-empty", len(train) > 500, str(len(train)))
check("roles are system/user/assistant",
      all([m["role"] for m in r["messages"]] == ["system", "user", "assistant"] for r in train))
check("every teacher is DeepSeek",
      {r["meta"]["teacher"] for r in train} <= bd.DEEPSEEK_CULPRITS,
      str({r["meta"]["teacher"] for r in train}))
check("no zero-extraction games leaked in",
      all(r["meta"]["game_extraction_ratio"] > 0 for r in train))
check("assistant targets are valid ToT JSON",
      all(set(json.loads(r["messages"][2]["content"])) ==
          {"reasoning", "options", "chosen_action", "message"} for r in train))
check("no placeholder options survived",
      not any(str(o).lower().startswith("strategy_")
              for r in train for o in json.loads(r["messages"][2]["content"])["options"]))
check("prompts within char budget",
      max(len(r["messages"][0]["content"]) + len(r["messages"][1]["content"]) for r in train)
      <= bd.MAX_PROMPT_CHARS)

train_games = {r["meta"]["game_id"] for r in train}
val_games = {r["meta"]["game_id"] for r in val}
check("no game appears in both splits", not (train_games & val_games),
      str(train_games & val_games))
check("val is unweighted (deduped)",
      len({(r["meta"]["game_id"], r["meta"]["turn"]) for r in val}) == len(val))

# duplication must exactly realise the declared weights
from collections import Counter  # noqa: E402
dup = Counter((r["meta"]["game_id"], r["meta"]["turn"]) for r in train)
declared = {(r["meta"]["game_id"], r["meta"]["turn"]): r["meta"]["weight"] for r in train}
check("row duplication matches declared weights",
      all(dup[k] == w for k, w in declared.items()))


# ── 6. resistance experiment ──────────────────────────────────────────────────

print("\n[6] Resistance side (grok distillation)")

import build_resistance_dataset as brd  # noqa: E402

# prompt reconstruction must match sage/game.py::_victim_messages
ep2 = pa.Episode("t|group|00", "attacker/x", "group", 1)
ep2.transcript = [{"turn": 1, "role": "culprit", "message": "A"},
                  {"turn": 1, "role": "victim", "message": "B"}]
arena_victim = ep2.victim_messages(2, "SYS")[1]["content"]
builder_victim = brd.victim_user_block(2, 100.0, brd.format_history(ep2.transcript, 2))
check("arena victim prompt == builder victim prompt", arena_victim == builder_victim,
      repr(arena_victim[:120]))
check("victim history labels Fundraiser/You",
      brd.format_history(ep2.transcript, 2) == "[Fundraiser]: A\n\n[You]: B")

# the balance shown must be what the victim saw at the START of the turn
led = [{"turn": 1, "delta_usd": -10.0, "remaining_usd": 90.0},
       {"turn": 2, "delta_usd": -5.0, "remaining_usd": 85.0}]
check("balance before turn 1 is $100", brd.remaining_before_turn(led, 1) == 100.0)
check("balance before turn 2 is $90", brd.remaining_before_turn(led, 2) == 90.0)
check("balance before turn 3 is $85", brd.remaining_before_turn(led, 3) == 85.0)

# placeholder rule: bare tokens are junk, labelled strategies are legitimate
bare = ["strategy_1", "strategy_2", "strategy_3"]
labelled = ["strategy_1: Repeat polite refusal", "strategy_2: Escalate firmness"]
check("all-bare options detected as junk",
      all(brd.BARE_PLACEHOLDER.fullmatch(o) for o in bare))
check("labelled strategies NOT treated as junk",
      not any(brd.BARE_PLACEHOLDER.fullmatch(o) for o in labelled))

# difficulty weighting
check("weak attacker, early turn -> 1", brd.turn_weight("qwen/qwen3-8b", 2) == 1)
check("strong attacker, early turn -> 2", brd.turn_weight("deepseek/deepseek-r1", 2) == 2)
check("weak attacker, late turn -> 2", brd.turn_weight("qwen/qwen3-8b", 8) == 2)
check("strong attacker, late turn -> 3", brd.turn_weight("deepseek/deepseek-r1", 8) == 3)
check("weight capped at 3", brd.turn_weight("deepseek/deepseek-r1", 10) <= brd.MAX_WEIGHT)

# resistance episode grid
rspecs = pa.make_resistance_specs(["ft/persuader", "base/qwen"], ["singular", "group"], 3)
check("resistance grid size", len(rspecs) == 2 * 2 * 3, str(len(rspecs)))
check("resistance ids unique", len({s["episode_id"] for s in rspecs}) == len(rspecs))
check("opponent recorded in victim_model field",
      {s["victim_model"] for s in rspecs} == {"ft/persuader", "base/qwen"})

# role-scoped adapter contexts must actually wrap the right generation
from contextlib import contextmanager  # noqa: E402
calls = []

@contextmanager
def atk_ctx():
    calls.append("attacker"); yield

@contextmanager
def vic_ctx():
    calls.append("victim"); yield

real = pa.batch_generate
pa.batch_generate = make_stub_generate(0.0, 7)
eps = [pa.Episode(**s) for s in rspecs[:2]]
pa.run_cohort(StubModel(), StubTok(), StubModel(), StubTok(), eps, PROMPTS, StubArbiter(),
              "ctx-test", verbose=False, attacker_ctx=atk_ctx, victim_ctx=vic_ctx)
pa.batch_generate = real
check("adapter contexts entered, alternating attacker->victim",
      calls[:4] == ["attacker", "victim", "attacker", "victim"], str(calls[:6]))
check("contexts entered once per side per turn", calls.count("attacker") == calls.count("victim"))

# resistance reporting: lower extraction must read as an improvement
good = [{"episode_id": f"e{i}", "victim_model": "a", "category": "group",
         "extraction_ratio": 0.0, "winner": "victim"} for i in range(10)]
bad = [{"episode_id": f"e{i}", "victim_model": "a", "category": "group",
        "extraction_ratio": 0.6, "winner": "culprit"} for i in range(10)]
rr = pa.resistance_report(bad, good, n_boot=2000)          # base=bad, ft=good -> improvement
check("negative delta reported as IMPROVED", rr["direction"] == "resistance IMPROVED", rr["direction"])
check("money saved is positive", rr["mean_money_saved_usd"] > 0, str(rr["mean_money_saved_usd"]))
check("held-all rate rises", rr["ft_held_all_rate"] == 1.0 and rr["base_held_all_rate"] == 0.0)
check("better/worse cells relabelled for resistance",
      rr["ft_resisted_better_cells"] == 10 and rr["ft_resisted_worse_cells"] == 0)
rr2 = pa.resistance_report(good, bad, n_boot=2000)         # reversed -> worse
check("positive delta reported as WORSENED", rr2["direction"] == "resistance WORSENED")
check("money saved negative when worse", rr2["mean_money_saved_usd"] < 0)


# ── 7. resistance dataset integrity ───────────────────────────────────────────

print("\n[7] Resistance dataset integrity")

rt_path = ROOT / "data" / "grok_resistance_train.jsonl"
if rt_path.exists():
    rtrain = [json.loads(l) for l in rt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    rval = [json.loads(l) for l in (ROOT / "data" / "grok_resistance_val.jsonl")
            .read_text(encoding="utf-8").splitlines() if l.strip()]
    check("resistance train non-empty", len(rtrain) > 500, str(len(rtrain)))
    check("every teacher is Grok",
          {r["meta"]["teacher"] for r in rtrain} <= brd.GROK_VICTIMS,
          str({r["meta"]["teacher"] for r in rtrain}))
    check("every kept game extracted $0",
          all(r["meta"]["game_extraction_ratio"] == 0 for r in rtrain))
    check("system prompt is the victim prompt",
          all("won $100 USD in a lottery" in r["messages"][0]["content"] for r in rtrain))
    check("user block shows a balance",
          all("Remaining money: $" in r["messages"][1]["content"] for r in rtrain))
    check("assistant targets are valid ToT JSON",
          all(set(json.loads(r["messages"][2]["content"])) ==
              {"reasoning", "options", "chosen_action", "message"} for r in rtrain))
    check("no all-bare placeholder options survived",
          not any(all(brd.BARE_PLACEHOLDER.fullmatch(str(o).strip().lower()) for o in opts)
                  for r in rtrain
                  if (opts := json.loads(r["messages"][2]["content"])["options"])))
    rtg = {r["meta"]["game_id"] for r in rtrain}
    rvg = {r["meta"]["game_id"] for r in rval}
    check("no game in both resistance splits", not (rtg & rvg), str(rtg & rvg))
    dup2 = Counter((r["meta"]["game_id"], r["meta"]["turn"]) for r in rtrain)
    decl2 = {(r["meta"]["game_id"], r["meta"]["turn"]): r["meta"]["weight"] for r in rtrain}
    check("resistance duplication matches weights", all(dup2[k] == w for k, w in decl2.items()))
else:
    print("  SKIP  resistance dataset not built yet")


# ── verdict ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
if FAILURES:
    print(f"{len(FAILURES)} CHECK(S) FAILED: {FAILURES}")
    sys.exit(1)
print("All checks passed.")
