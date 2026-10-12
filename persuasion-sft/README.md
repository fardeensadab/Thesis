# persuasion-sft — distilling DeepSeek's persuasion policy into a small Qwen

**Question:** the SAGE tournament (`the_game`) found `deepseek/deepseek-r1` to be the strongest
persuader of 15 models. Can that skill be *transferred* — does supervised distillation of DeepSeek's
winning transcripts measurably raise a small open model's persuasive power above its own base?

This repo builds the dataset, fine-tunes the student on Kaggle, and answers that question with a
paired experiment scored by the tournament's own arbiter.

---

## 1. Where the teachers came from

Recomputed over all 1,049 completed tournament games (`extraction_ratio` = fraction of the victim's
$100 talked out of them over ≤10 turns):

| rank | culprit (persuader) | n | mean extraction | win rate |
|---|---|---:|---:|---:|
| **1** | **deepseek/deepseek-r1** | 70 | **0.319** | 24.3% |
| 2 | x-ai/grok-4 | 70 | 0.265 | 18.6% |
| 3 | google/gemini-2.5-flash | 70 | 0.262 | 15.7% |
| 4 | anthropic/claude-opus-4.6 | 70 | 0.246 | 17.1% |
| **5** | **deepseek/deepseek-v3.2** | 70 | **0.226** | 17.1% |
| … | | | | |
| 15 | qwen/qwen3-8b | 70 | 0.091 | 8.6% |

Both DeepSeek variants are the teachers. Their 140 games are the raw material.

## 2. The dataset

`scripts/build_dataset.py` reconstructs each culprit turn as a chat example whose system and user
strings are **byte-identical** to `sage/game.py::_culprit_messages`, with the full Tree-of-Thought
JSON (`reasoning` / `options` / `chosen_action` / `message`) as the target. A model trained on this
drops straight back into the arena with no protocol change — which is what makes the base and tuned
arms scoreable by identical machinery.

Selection is rejection-sampling style, since cloning a policy means cloning the *successful* part:

| step | effect |
|---|---|
| DeepSeek-as-culprit games | 140 |
| drop games that extracted $0 | −57 → **83 games** (mean ratio 0.459) |
| drop schema-placeholder ToT (`strategy_1`…) | −16 turns |
| drop victim repetition-loop contamination (>12k chars) | −11 turns |
| **unique training turns** | **595** |
| weighted rows (`1 + [paid this turn] + [game ratio ≥ 0.30]`, capped at 3) | **1,073** |
| validation rows (split **by game**, never duplicated) | **65** |

Splitting by game rather than by turn matters: turn-level splitting would put nearly the whole
conversation on both sides of the boundary and make validation loss meaningless.

## 3. The experiment

| | |
|---|---|
| student | `Qwen/Qwen3-4B-Instruct-2507` (QLoRA, 4-bit NF4, r=32, all-linear) |
| victims | Qwen2.5-1.5B-Instruct, SmolLM2-1.7B-Instruct, Phi-3.5-mini-instruct, TinyLlama-1.1B-Chat |
| judge | `x-ai/grok-4.1-fast`, 2-of-3 consensus — the tournament's own arbiter and prompt |
| design | paired: both arms run the same (victim × category × seed) grid |
| statistics | paired mean delta, bootstrap 95% CI, sign-flip permutation p-value |

Three design choices do the real work:

* **One set of weights.** The adapter is attached to the 4-bit base and switched off with
  `disable_adapter()` for the base arm. The arms cannot differ by quantisation, kernel or sampler.
* **Paired cells.** Persuasion outcomes are high-variance and bounded; unpaired means over ~50
  episodes would bury a real effect. This is the flaw that left the earlier GRPO ablation unable to
  distinguish "no learning" from "not enough episodes".
* **Cohort batching.** Episodes advance in lockstep and generate as one padded batch, so wall-clock
  scales with turns rather than episodes — which is what makes ~100 ten-turn episodes fit in a
  Kaggle session.

## 4. Layout

```
persuasion-sft/
├── scripts/
│   ├── build_dataset.py      extract + filter + weight the DeepSeek turns
│   ├── make_notebooks.py     generate both .ipynb from source
│   ├── test_arena.py         41 offline checks — no GPU, no API key
│   └── push_to_kaggle.py     upload dataset + notebooks
├── data/                     the generated dataset + its card
├── kaggle_dataset/           exactly what is uploaded (arena code, prompts, jsonl)
│   └── persuasion_arena.py   the environment: prompts, ToT parser, arbiter, ledger, statistics
└── notebooks/
    ├── 01_train_sft.ipynb    QLoRA distillation
    └── 02_eval_arena.ipynb   paired base-vs-tuned arena
```

## 5. Running it

See **[SETUP.md](SETUP.md)**. Short version: the dataset and both notebooks are already on Kaggle —
add your `OPENROUTER_API_KEY` secret, run 01, wire its output into 02, run 02.

Locally:

```bash
python scripts/build_dataset.py \
  --games-dir  "I:/IUT CSE/Thesis/NLP/Implementation/the_game/games" \
  --prompts-dir "I:/IUT CSE/Thesis/NLP/Implementation/the_game/prompts" \
  --out-dir data
python scripts/test_arena.py      # 41 checks, runs in seconds on CPU
python scripts/make_notebooks.py
```

## 6. Interpreting the result

The number to report is the **paired delta in extraction ratio** with its CI and p-value.

* **Delta > 0, CI excludes 0** — persuasive skill transferred by imitation alone. Check the
  per-victim table to see whether the gain is broad or driven by one soft target.
* **Delta ≈ 0, tight CI** — a genuine null, and a far sharper one than the GRPO ablation produced.
  Before reporting it, confirm the base arm is not already near ceiling on these victims; if it is,
  there was no headroom, and the fix is the skeptical victim prompt from §9 of the experiment
  report, not more training.
* **Delta < 0** — cloning 595 turns of a much larger teacher pushed the 4B off its own
  distribution. Try fewer epochs or a lower LoRA rank before concluding anything.

One caveat belongs in the write-up either way: these victims are far smaller than the tournament's
Claude/GPT/Grok pool, so absolute extraction ratios are **not** comparable to the 15-model
leaderboard above. The paired delta is the quantity this design measures.
