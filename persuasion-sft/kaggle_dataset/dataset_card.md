# DeepSeek Persuasion SFT set

Distilled from the SAGE `the_game` tournament: every culprit (persuader) turn played by
`deepseek/deepseek-r1` or `deepseek/deepseek-v3.2` in a game that extracted more than $0.

## Provenance

- DeepSeek-as-culprit games in tournament: **140**
- Dropped (extracted $0): **57**
- Games kept: **83** (mean extraction ratio 0.459, median 0.300)
- Dropped degenerate turns (empty / fallback / <40 chars): **0**
- Dropped turns with no recorded reasoning: **0**
- Dropped turns whose ToT options were schema placeholders (`strategy_1`...): **16**
- Dropped turns whose prompt exceeded 12,000 chars (victim repetition-loop contamination): **11**

## Splits (split by game, so no dialogue prefix crosses the boundary)

- train: **1073** rows after weighting (**595** unique turns)
- val: **65** rows (unweighted)

## Weighting

`weight = 1 + [victim paid on this turn] + [game extraction ratio >= 0.30]`, capped at 3, realised by row duplication in the train split only.

| weight | unique turns |
|---|---|
| 1 | 291 |
| 2 | 195 |
| 3 | 174 |

## Teacher mix (unique turns)

| model | turns |
|---|---|
| deepseek/deepseek-v3.2 | 365 |
| deepseek/deepseek-r1 | 295 |

## Format

Each row is `{"messages": [system, user, assistant], "meta": {...}}`.
The `system` and `user` strings are reconstructed to be byte-identical to
`sage/game.py::_culprit_messages`, and the `assistant` string is the Tree-of-Thought JSON
the tournament arbiter pipeline expects. Train on the assistant turn only.
