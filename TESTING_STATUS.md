# TetrisWeiqi Testing Status

Last updated: 2026-04-15

## Current Go-Style Mainline Rules

The current recommended mainline rules are:

1. Board size: `10x10`
2. Piece distribution: shared `bag7`
   Chinese: shared seven-bag piece distribution
3. Terminal scoring: `pieces_only`
   Chinese: final result compares only surviving piece counts
4. Voluntary skip: disabled
5. End condition: `double_forced_pass`
   Chinese: the game ends after two consecutive forced passes
6. No-legal-move handling: `reroll_once_then_pass`
   Chinese: reroll once when no legal move exists, then forced pass if still blocked
7. Resolution order: `capture_then_clear_once`
   Chinese: capture first, then line clear, with no second capture recheck
8. Capture style: Go-style removal to `EMPTY`
   Chinese: surrounded groups are removed directly instead of becoming dead zones
9. Local legality search: enabled
   Chinese: optimized local search path is enabled for move legality and settlement checks

## Mainline Direction Change

The old dead-zone mainline is now archived as a future variant idea.

The current mainline is the newer Go-style ruleset:

1. Surrounded groups are removed directly.
2. Dead-zone occupation/conversion is not part of the current mainline.
3. Existing dead-zone-oriented terminal modes are no longer meaningful differentiators under the mainline.

## Rule Testing Summary Under The Current Mainline

### Kept

1. `bag7` stayed on the mainline over `bag7_independent`.
2. `pieces_only` stayed as the practical terminal objective.
3. `reroll_once_then_pass` stayed preferred over `pass_and_redraw`.
4. `capture_then_clear_once` became the preferred resolution order.
5. Voluntary `skip` remains disabled.
6. Board size remains `10x10`.

### Archived Or Deprioritized

1. Dead-zone scoring and dead-zone conversion are archived with the old dead-zone variant line.
2. `pieces_then_deadzones` and the old `area_like` mode are not meaningful mainline candidates under Go-style capture.
3. `pass_and_redraw` was dropped for the current mainline.
4. `capture_then_clear_recheck` was dropped from mainline priority after Go-style testing.
5. `bag7_independent` was dropped from mainline priority.

## Current Rule Experiment Results

### No-Legal-Move Handling

Compared:

1. `reroll_once_then_pass`
   Chinese: reroll once, then forced pass
2. `pass_and_redraw`
   Chinese: pass immediately and redraw for the next turn

Heuristic screening with `200` games:

1. `reroll_once_then_pass`
   - First-player winrate: `55.0%`
   - Average moves: `47.9`
   - Average line clears: `9.2`
2. `pass_and_redraw`
   - First-player winrate: `64.5%`
   - Average moves: `33.7`
   - Average line clears: `4.6`

Short training result:

1. `reroll_once_then_pass`: best eval winrate `37.5%`
2. `pass_and_redraw`: best eval winrate `30.0%`

Conclusion:
Keep `reroll_once_then_pass`.

### Resolution Order

Compared:

1. `capture_then_clear_recheck`
   Chinese: capture first, clear lines second, then capture again if needed
2. `clear_then_capture`
   Chinese: clear lines first, then capture
3. `capture_then_clear_once`
   Chinese: capture first, clear lines second, no second capture recheck

Heuristic screening with `200` games:

1. `capture_then_clear_recheck`
   - First-player winrate: `55.0%`
   - Average moves: `47.9`
2. `clear_then_capture`
   - First-player winrate: `49.5%`
   - Average moves: `53.0`
3. `capture_then_clear_once`
   - First-player winrate: `55.0%`
   - Average moves: `47.9`

Short training result:

1. `capture_then_clear_recheck`: best eval winrate `22.5%`
2. `clear_then_capture`: best eval winrate `35.0%`
3. `capture_then_clear_once`: best eval winrate `42.5%`

Conclusion:
Keep `capture_then_clear_once` as the current mainline candidate.

### Piece Distribution

Compared:

1. `bag7`
   Chinese: shared seven-bag distribution
2. `bag7_independent`
   Chinese: each player uses an independent seven-bag distribution

Heuristic screening with `200` games:

1. `bag7`
   - First-player winrate: `55.0%`
   - Average moves: `47.9`
   - Average line clears: `9.2`
2. `bag7_independent`
   - First-player winrate: `60.0%`
   - Average moves: `44.8`
   - Average line clears: `8.3`

Short training result:

1. `bag7`: best eval winrate `45.0%`
2. `bag7_independent`: best eval winrate `15.0%`

Conclusion:
Keep shared `bag7`.

## Performance And Safety Status

### Rule Engine Optimization

The current engine includes local-search and candidate-mark optimizations for:

1. legal move generation
2. capture checks
3. line clear checks

These optimizations significantly improved training throughput in local testing.

### Consistency Guardrail

Rule-consistency validation script:

`cli/check_rule_consistency.py`

Recommended command:

```powershell
python cli\check_rule_consistency.py --states 100 --move-checks 3 --seed 20260413
```

This script compares the optimized engine against a more conservative reference implementation on random states and sampled moves.

## Current Recommended Training Baseline

Recommended training baseline:

1. `piece_distribution = bag7`
2. `terminal_mode = pieces_only`
3. `allow_voluntary_skip = false`
4. `end_condition_mode = double_forced_pass`
5. `no_legal_move_mode = reroll_once_then_pass`
6. `resolution_mode = capture_then_clear_once`
7. `board_size = 10`

Recommended 4070 Ti SUPER training profile under the current Go-style mainline:

1. `games-per-iter = 24`
2. `num-simulations = 16`
3. `selfplay-parallel-games = 10`
4. `inference-batch-size = 40`
5. `batch-size = 512`
6. `lr = 0.0015`
7. `lr-step-size = 15`
8. `lr-gamma = 0.9`
9. `dirichlet-alpha = 0.05`
10. `dirichlet-epsilon = 0.10`
11. `train-steps-per-iter = 16`

Chinese summary:
This profile is the current post-fix stable baseline. It keeps the corrected
search implementation, uses light root exploration noise, and avoids the less
stable combinations that stacked higher simulations and higher train steps at
the same time.

## Training Parameter Retuning Under Go-Style Mainline

Three short directional experiments were run under the fixed mainline rules:

1. `checkpoints_tune_go_baseline_short`
   Chinese: current baseline short run
2. `checkpoints_tune_go_more_updates`
   Chinese: same self-play data, but more optimizer updates per iteration
3. `checkpoints_tune_go_more_data`
   Chinese: more self-play games per iteration, plus a moderate fixed number of updates

Short-run result summary:

1. Baseline short
   - config: `24 games/iter`, `24 sims`, `min_train_batches=16`, `lr=0.002`
   - best eval winrate: `32.5%`
2. More updates
   - config: `24 games/iter`, `24 sims`, `train_steps_per_iter=32`, `lr=0.0015`
   - best eval winrate: `35.0%`
3. More data
   - config: `36 games/iter`, `24 sims`, `train_steps_per_iter=24`, `lr=0.0015`
   - best eval winrate: `37.5%`

Interpretation:

1. The current Go-style mainline was not fully saturated yet.
2. There is real optimization space on the training side without changing rules.
3. Adding more fresh self-play data each iteration currently helps more than only increasing training steps.
4. A slightly lower learning rate also appears to reduce instability compared with the old `0.002` baseline.

Additional cross-check:

1. `checkpoints_tune_go_more_data/best.pt` vs `checkpoints_mainline_c_v2/best.pt`
   - as first player set: `42.5%` winrate with `17` draws in `80` games
2. reverse seat order:
   - current mainline best achieved `33.8%` winrate in `80` games

This head-to-head does not fully eliminate variance, but it supports promoting
the `more_data` profile as the new training default candidate.

## Long Training Result With Updated Mainline Profile

A longer run was completed with the updated 4070 Ti SUPER profile in:

`checkpoints_mainline_c_v2`

Evaluation checkpoints:

1. Iteration `5`: `35.0%` heuristic winrate (`42-74-4`)
2. Iteration `10`: `41.7%` heuristic winrate (`50-61-9`)
3. Iteration `15`: `41.7%` heuristic winrate (`50-62-8`)
4. Iteration `20`: `43.3%` heuristic winrate (`52-62-6`)

Current takeaway:

1. The updated training profile materially improved the Go-style mainline.
2. The current Go-style long run now clearly exceeds the previous Go-style run that peaked around `34.2%`.
3. The model was still improving at iteration `20`, so this line does not look saturated yet.

## Resume Semantics And Checkpoint Fix

Training checkpoint handling was updated in `cli/train_alphazero.py`:

1. checkpoints now save `scheduler` state
2. checkpoints now save `best_model_state`
3. `resume` now restores scheduler state when available
4. `resume` now prints a warning that replay buffer is not restored

Interpretation:

1. old `resume` runs were warm-start continuations, not strict seamless continuations
2. so earlier `20 -> 40` resume-based degradation should not be over-interpreted as pure model regression

## Fresh 40-Iteration Continuous Run

A fully fresh `40`-iteration run was also completed without `resume`:

`checkpoints_mainline_c_v3_fresh40`

Evaluation checkpoints:

1. Iteration `20`: `40.0%`
2. Iteration `30`: `41.7%`
3. Iteration `40`: `32.5%`

Interpretation:

1. under a continuous run, performance can still improve after `20`
2. but longer training does not reliably keep improving all the way through `40`
3. the current bottleneck looks more like mid/late training stability than raw training length

## Search Correctness And Exploration Fixes

The training stack was updated again to fix three quality-critical issues:

1. MCTS game clones now copy RNG state correctly
2. self-play root nodes now support Dirichlet exploration noise
3. legal-move priors now fall back to a uniform distribution when the summed legal prior collapses to zero

Interpretation:

1. the old search path could diverge from the real environment after future piece draws or rerolls
2. the new path is more correct, but also changes the effective training dynamics
3. so post-fix training curves should be compared carefully against pre-fix curves

## Dirichlet Noise Strength Tuning

Three short post-fix training tests were run to tune root exploration noise:

1. `checkpoints_noise_std_short`
   - config: `dirichlet-alpha=0.3`, `dirichlet-epsilon=0.25`
   - best eval winrate: `40.0%`
2. `checkpoints_noise_mid_short`
   - config: `dirichlet-alpha=0.15`, `dirichlet-epsilon=0.15`
   - best eval winrate: `28.3%`
3. `checkpoints_noise_light_short`
   - config: `dirichlet-alpha=0.05`, `dirichlet-epsilon=0.10`
   - best eval winrate: `45.0%`

Conclusion:

1. root exploration noise should be kept
2. the earlier standard AlphaZero-style noise was too strong for the current game
3. a lighter noise setting currently looks like the best default

## 20-Iteration Long Run With Light Root Noise

A full `20`-iteration run was completed after the search fixes and light-noise update:

`checkpoints_mainline_c_v2`

Evaluation checkpoints:

1. Iteration `5`: `29.2%` heuristic winrate (`35-75-10`)
2. Iteration `10`: `30.8%` heuristic winrate (`37-72-11`)
3. Iteration `15`: `30.0%` heuristic winrate (`36-80-4`)
4. Iteration `20`: `34.2%` heuristic winrate (`41-73-6`)

Interpretation:

1. lighter noise is better than the heavier post-fix noise settings
2. but the fully corrected training stack has not yet reproduced the older pre-fix `43.3%` peak
3. the project is now on a more trustworthy search implementation, but additional training retuning is still needed

## Post-Fix Retuning Summary

Several post-fix retuning sweeps were run under the corrected search stack.

Short `10`-iteration comparison:

1. `checkpoints_postfix_baseline`
   - config: `24 games`, `16 sims`, `16 train_steps`
   - best eval winrate: `40.0%`
2. `checkpoints_postfix_more_data`
   - config: `36 games`, `16 sims`, `16 train_steps`
   - best eval winrate: `30.0%`
3. `checkpoints_postfix_more_search`
   - config: `24 games`, `24 sims`, `16 train_steps`
   - best eval winrate: `41.7%`
4. `checkpoints_postfix_more_train`
   - config: `24 games`, `16 sims`, `24 train_steps`
   - best eval winrate: `41.7%`

Interpretation:

1. simply adding more self-play games per iteration is not the current main direction
2. stronger search or more train steps can help in short runs
3. but stacking both increases together in a formal long run did not stay stable enough to become the new default

Formal `20`-iteration combination check:

1. `24 games`, `24 sims`, `24 train_steps`
   - best eval winrate: `33.3%`

So the project currently keeps the lighter post-fix baseline as the safer
default, while larger search/training combinations remain experimental.

## Evaluation Caveat

Additional layered evaluation was used to sanity-check whether the model is
really becoming stronger in a human-meaningful sense.

Using `checkpoints_postfix_baseline/best.pt`:

1. vs pure random: `35.0%`
2. vs weak random + heuristic mix: about `46.7%` to `47.5%`
3. vs `heuristic_lv1`: `56.7%`
4. vs `heuristic_lv2`: `28.3%`

Interpretation:

1. the current model should not yet be described as clearly stronger than random play
2. the existing heuristic baselines are useful, but they are not sufficient as the only long-term yardstick
3. future training should prioritize strategy growth and self-improvement over chasing a single benchmark number

## Long-Term Training Direction

The long-term goal is not merely to beat random or exploit one heuristic AI.
The goal is a model that can face humans, keeps improving over time, and shows
clearer strategic behavior.

This changes the preferred direction:

1. keep the corrected search stack and the stable post-fix baseline
2. treat `vs_random` only as a sanity check, not the final target
3. strengthen evaluation around self-improvement, fixed old checkpoints, and strategy probes
4. prioritize whether the model is learning capture timing, defense, shape, and tradeoffs, not only headline winrate

## Mid-Late Training Stability Tuning

Three controlled `30`-iteration training-strategy tests were run:

1. `checkpoints_tune_mid_baseline30`
   - config: `lr-step-size=10`, `lr-gamma=0.8`, `buffer-size=100000`
   - best eval winrate: `38.8%`
2. `checkpoints_tune_mid_slowdecay30`
   - config: `lr-step-size=15`, `lr-gamma=0.9`, `buffer-size=100000`
   - best eval winrate: `41.2%`
3. `checkpoints_tune_mid_smallbuffer30`
   - config: `lr-step-size=10`, `lr-gamma=0.8`, `buffer-size=30000`
   - best eval winrate: `36.2%`

Conclusion:

1. a slower and smoother learning-rate decay helps current Go-style training more than shrinking the replay buffer
2. smaller replay buffer is not the current main optimization direction
3. the current recommended training default should move to the slower decay profile

## Cross-Check Against The Older Representative Model

To avoid over-reading heuristic-only gains, the new Go-style best model was also
compared against the preserved older representative checkpoint:

1. New Go-style best vs old representative
   - `46-57-17` in `120` games
   - winrate: `38.3%`
2. Old representative vs new Go-style best
   - `62-41-17` in `120` games
   - winrate: `51.7%`
3. New Go-style best vs heuristic Lv2
   - `42-69-9`
   - winrate: `35.0%`
4. Old representative vs heuristic Lv2
   - `47-69-4`
   - winrate: `39.2%`

Interpretation:

1. The new training profile improved the Go-style mainline itself.
2. But the current Go-style best still does not consistently beat the preserved older representative model.
3. So the mainline training setup should be considered improved, while overall model replacement is not yet fully justified.

## Next Suggested Step

The next useful step is to upgrade evaluation from a single benchmark into a
multi-axis framework, then continue training from the stable corrected
baseline rather than chasing one short-term winrate number.
