# Grok Resistance SFT set

Distilled from the SAGE `the_game` tournament: every victim (resistor) turn played by
`x-ai/grok-4.1-fast` or `x-ai/grok-4` in a game where the attacker extracted **$0**.

## Provenance

- Grok-as-victim games in tournament: **140**
- Dropped (attacker extracted > $0): **2**
- Games kept: **138** (all held the full $100)
- Dropped degenerate turns: **74**
- Dropped turns with no recorded reasoning: **0**
- Dropped placeholder-ToT turns: **211**
- Dropped oversized prompts: **30**

## Splits (by game, so no dialogue prefix crosses the boundary)

- train: **1486** rows after weighting (**842** unique turns)
- val: **121** rows (unweighted)

## Weighting

Grok surrenders no money, so there is no outcome gradient to rank turns by. Difficulty is
used instead: `weight = 1 + [attacker is a top-5 persuader] + [turn >= 6]`, capped at 3,
realised by row duplication in the train split only.

| weight | unique turns |
|---|---|
| 1 | 374 |
| 2 | 455 |
| 3 | 134 |

## Teacher mix (unique turns)

| model | turns |
|---|---|
| x-ai/grok-4.1-fast | 538 |
| x-ai/grok-4 | 425 |

## Attackers resisted (unique turns)

| attacker | turns | top-5 |
|---|---|---|
| qwen3-ft:latest | 86 |  |
| openai/gpt-5.4 | 78 |  |
| google/gemma-3-4b-it | 77 |  |
| google/gemini-2.5-flash | 76 | yes |
| deepseek/deepseek-r1 | 76 | yes |
| meta-llama/llama-4-maverick | 75 |  |
| mistralai/mistral-nemo | 73 |  |
| google/gemini-3-flash-preview | 70 |  |
| deepseek/deepseek-v3.2 | 68 | yes |
| qwen/qwen3-max | 64 |  |
| deepseekR1-ft:latest | 60 |  |
| anthropic/claude-opus-4.6 | 57 | yes |
| x-ai/grok-4.1-fast | 44 |  |
| x-ai/grok-4 | 36 | yes |
| qwen/qwen3-8b | 23 |  |

## Format

Each row is `{"messages": [system, user, assistant], "meta": {...}}`. The `system` and
`user` strings are reconstructed to be byte-identical to `sage/game.py::_victim_messages`
(including the running balance the victim saw), and the `assistant` string is the
Tree-of-Thought JSON. Train on the assistant turn only.
