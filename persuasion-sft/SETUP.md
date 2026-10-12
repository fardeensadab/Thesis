# Kaggle setup

Everything is already uploaded to your account. Three things to do before the first run.

## What is live

| artefact | link |
|---|---|
| Dataset (arena code, prompts, SFT jsonl) | https://www.kaggle.com/datasets/fardeensadab/deepseek-persuasion-sft |
| Notebook 01 — distillation | https://www.kaggle.com/code/fardeensadab/persuasion-01-distil-deepseek-into-qwen3-4b |
| Notebook 02 — arena | https://www.kaggle.com/code/fardeensadab/persuasion-02-arena-base-vs-fine-tuned |

All three are **private**. The dataset is already attached to both notebooks.

---

## ⚠ The tournament's arbiter model is dead

`x-ai/grok-4.1-fast` — the judge in `the_game/sage/config.py` — has been **deprecated by xAI**.
OpenRouter now answers every request to it with a 404.

This matters far beyond this repo. `sage/arbiter.py::_call_arbiter` catches all exceptions and
returns `'{"given_usd": 0, ...}'`, so a dead model does not crash the tournament — it makes every
arbiter call return **$0**. Every episode scores zero extraction, and the output looks like a
clean, publishable null result instead of a broken judge. **If you re-run `the_game`, change
`ARBITER_MODEL` first.**

The arena here now uses **`x-ai/grok-4.3`** (xAI's recommended successor, and the same judge as
the persuasion-RL ablation, so the two experiments stay consistent). Verified scoring correctly on
known cases: `$15 → 15.0`, `$40 → 40.0`, refusal `→ 0.0`. A 404 from the arbiter now raises
loudly instead of falling through to $0, and `arbiter.n_failures` counts any call that had to
fall back — it must be 0 for a run to be valid.

## Step 1 — add the OpenRouter secret (notebook 02 only)

The arbiter that counts money runs via OpenRouter.

Open notebook 02 → **Add-ons → Secrets → Add a new secret**

* Label: `OPENROUTER_API_KEY`
* Value: the key from `the_game/.env`

Toggle it on for the notebook. Notebook 02 probes the arbiter with three known cases and stops
immediately if it is misconfigured — otherwise a bad key silently produces all-zero extraction and
looks like a null result.

## Step 2 — run notebook 01

Internet **on**. Accelerator: **GPU T4 x2** is the fast path, but the notebook now works on
either card — see below. Only GPU 0 is used.

**Save Version → Save & Run All (Commit)**, so the adapter is persisted as a notebook output.
Roughly 30–45 minutes: ~400 optimiser steps over 1,073 rows.

Output: `/kaggle/working/qwen3-4b-deepseek-persuader/final` (~130 MB of LoRA weights).

## Step 3 — wire the adapter into notebook 02, then run it

In notebook 02: **+ Add Input → Notebook Output →
`persuasion-01-distil-deepseek-into-qwen3-4b`**.

No path to type. Notebook 02 locates the adapter by searching for `adapter_config.json` under
`/kaggle/input`, and the dataset by searching for `persuasion_arena.py` — see the mount-path note
below for why hardcoding either would be fragile.

Then **Save & Run All (Commit)**. At the defaults (4 victims × 2 categories × 6 episodes = 48
episodes per arm, 96 total) expect **2–3 hours**.

For a first sanity pass, edit the config cell to `N_PER_CELL = 2` and a single victim — about
20 minutes, enough to confirm the plumbing before spending real GPU quota.

Outputs land in `/kaggle/working/arena_results/`: `summary.json`, `episodes.csv`,
`episodes_base.json`, `episodes_finetuned.json`, `arena_summary.png`.

---

## Cost and quota

* **GPU:** ~3–4 h total, against Kaggle's 30 h/week free allowance.
* **OpenRouter:** ~2,000 `grok-4.1-fast` calls (2 per turn, plus a tie-breaker on disagreement),
  each a short JSON extraction. Well under a dollar at current pricing, but the notebook prints the
  running call count so you can watch it.

## Re-pushing after local edits

```bash
export KAGGLE_API_TOKEN=KGAT_...      # NOT KAGGLE_KEY — see the note below
export KAGGLE_USERNAME=fardeensadab
export KAGGLE_KEY=unused

python scripts/make_notebooks.py
python scripts/push_to_kaggle.py              # dataset + both notebooks
python scripts/push_to_kaggle.py --kernels    # notebooks only
```

## GPU note (why runs 2 and 3 failed)

Pushing a notebook through the Kaggle API sets `enable_gpu: true`, which resolves to the legacy
default: a **Tesla P100 (sm_60)**. Kaggle's image ships torch 2.10, which dropped Pascal — its
kernels cover sm_70 and up — so on a P100 the 4-bit bitsandbytes calls abort with
`named symbol not found … ops.cu` and take the Jupyter kernel with them.

The accelerator **cannot be set through the API** (the save-kernel request exposes only
`enable_gpu` / `enable_tpu`), and re-pushing a notebook resets whatever was chosen in the UI. So
rather than depend on a manual toggle, the preflight cell detects the card and adapts:

| card | what the preflight does |
|---|---|
| P100 (sm_60) | removes torchvision/torchaudio, installs `torch==2.7.1+cu126` — the last build shipping sm_60 kernels — then the ML stack. Adds ~3 min. |
| T4 / newer | pins torch to the image version so pip cannot swap the CUDA build, then installs the ML stack. |

The next cell asserts the running torch actually has kernels for the card's arch and runs a real
CUDA matmul, so an incompatible stack fails in one second rather than ten minutes in.

Verified on a live P100 run of the real notebook cells: torch 2.7.1 installs, arch list becomes
`['sm_50','sm_60','sm_70','sm_75','sm_80','sm_86','sm_90']`, the CUDA smoke test passes, the
dataset resolves, 1,073/65 rows tokenise with 0 skipped, and the 4-bit Qwen3-4B + LoRA loads in
**3.73 GB** with 66 M trainable parameters. A separate probe confirmed 4-bit forward+backward,
`generate()` and `PagedAdamW8bit` all work on Pascal.

One subtlety the P100 run exposed: `torch.cuda.is_bf16_supported()` returns **True** on sm_60
because recent versions count emulation. The notebooks therefore gate bf16 on `sm_80+` and fall
back to fp16 on both P100 and T4, instead of silently picking a dtype the hardware has to fake.

## Mount-path note (why the first run failed)

Kaggle's current image mounts **API-attached** inputs at

```
/kaggle/input/datasets/<owner>/<slug>/…
```

not at the legacy `/kaggle/input/<slug>/…` that UI-attached inputs and older images use. Both
notebooks therefore resolve their inputs by searching for a marker file
(`deepseek_persuasion_train.jsonl`, `persuasion_arena.py`, `adapter_config.json`) rather than
assuming a layout, and print the resolved directory. If an input is genuinely missing they raise a
`FileNotFoundError` listing what *is* mounted, instead of a bare "no such file".

Verified on a live Kaggle run: the dataset resolves to
`/kaggle/input/datasets/fardeensadab/deepseek-persuasion-sft`, 1,073 train / 65 val rows load, and
tokenisation keeps all of them (max 2,562 tokens, nothing truncated).

## Auth

**A `KGAT_…` token is not the legacy `kaggle.json` key.** Put it in `KAGGLE_KEY` and
every read returns 200 (public endpoints need no auth at all) while every write 401s. It has to go
in `KAGGLE_API_TOKEN`, which makes `kagglesdk` send `Authorization: Bearer`. The two legacy vars
still have to be set to something, because `KaggleApi.authenticate()` checks for them before the
HTTP layer ever gets a chance to use the bearer token.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `named symbol not found … ops.cu`, then "Kernel died" | old notebook version on a P100 | use version 6+; the preflight now installs a Pascal-compatible torch automatically |
| `torch … has no kernels for sm_XX` | preflight installed a new torch but the old one was already imported | Run → **Restart & Run All** |
| Notebook spends ~3 min on "pip install" at the start | it landed on a P100 and is fetching torch 2.7.1 | expected; switch to T4 x2 to skip it |
| All extraction ratios are 0.0 | arbiter key missing or wrong | the Step-1 probe cell will have failed — fix the secret |
| CUDA OOM in notebook 02 | Phi-3.5-mini is the largest victim | lower `COHORT_SIZE` to 4, or drop Phi from `VICTIMS` |
| `FileNotFoundError` naming a marker file | that input is not attached | the error lists what *is* mounted — add the missing input |
| `ADAPTER_PATH` not found | notebook 01 was run interactively, not committed | re-run it with Save & Run All |
| Tokenisation "skipped" count > 0 in 01 | chat template changed upstream | check the assertion in the encode cell before training |
| Base arm already extracts most of the $100 | ceiling effect — no headroom to improve | give victims the skeptical system prompt (experiment report §9) |
