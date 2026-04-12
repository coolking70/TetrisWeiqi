# TetrisWeiqi Testing Status

Last updated: 2026-04-12

## Current Mainline Rules

The current recommended mainline rules are:

1. Board size: `10x10`
2. Piece distribution: shared `bag7`
3. Dead zones count as filled cells for line clears
4. Terminal scoring: `pieces_only`
5. Voluntary skip: disabled
6. End condition: `double_forced_pass`
7. No-legal-move handling: `reroll_once_then_pass`
8. Resolution order: `capture_then_clear_recheck`
9. Dead-zone conversion activation: `immediate`
10. No-legal-move rerolls: `1`

## Rule Testing Summary

### Kept

1. `bag7` stayed on the mainline after bag-state encoding was added to the network input.
2. `pieces_only` stayed as the terminal mode. It remained the cleanest and safest objective.
3. `reroll_once_then_pass` replaced the older no-legal-move handling and became the current mainline.
4. `capture_then_clear_recheck` stayed as the preferred resolution order.
5. Dead-zone activation stayed `immediate`.
6. Shared `bag7` stayed preferred over independent per-player bags.

### Dropped or Deprioritized

1. Dead-zone score weight optimization was dropped to avoid numeric complexity.
2. Voluntary `skip` was dropped and is now disabled by default.
3. Terminal-mode variants such as `pieces_then_deadzones` and `area_like` did not beat `pieces_only`.
4. `single_forced_pass` and the original `pass_and_redraw` line both lost to the current mainline package.
5. Increasing no-legal-move rerolls from `1` to `2` made games longer and less balanced.
6. Dead-zone activation `next_turn` did not show a strong enough signal to justify changing the mainline.

## Main Rule Experiments

### A/B/C Endgame Package Comparison

These were tested under the fixed core rule set.

1. `A_baseline`
   `double_forced_pass + pass_and_redraw`
2. `B_strict_end`
   `single_forced_pass + pass_and_redraw`
3. `C_reroll_buffer`
   `double_forced_pass + reroll_once_then_pass`

Reliable long-training comparison results:

1. `A_baseline`: best heuristic eval winrate `31.7%`
2. `B_strict_end`: best heuristic eval winrate `33.3%`
3. `C_reroll_buffer`: best heuristic eval winrate `40.8%`

Conclusion:
`C_reroll_buffer` became the new mainline.

### Resolution Order Comparison

Heuristic screening and short training were run for:

1. `capture_then_clear_recheck`
2. `clear_then_capture`
3. `capture_then_clear_once`

Short training result:

1. `capture_then_clear_recheck`: best eval winrate `48.8%`
2. `clear_then_capture`: best eval winrate `46.2%`

Conclusion:
Keep `capture_then_clear_recheck`.

### Dead-Zone Activation Timing

Compared:

1. `immediate`
2. `next_turn`

Heuristic comparison showed only very small differences.

Conclusion:
Keep `immediate`.

### No-Legal-Move Rerolls

Compared:

1. `1` reroll
2. `2` rerolls

Heuristic comparison showed that `2` rerolls produced longer games and worse balance.

Conclusion:
Keep `1` reroll.

### Piece Distribution Symmetry

Compared:

1. shared `bag7`
2. `bag7_independent`

Heuristic comparison showed that shared `bag7` kept richer interaction and remained preferred.

Conclusion:
Keep shared `bag7`.

## Training and Evaluation Status

### Latest Mainline Long Training

Directory:
`checkpoints_mainline_c_v2`

This run used:

1. `24` self-play games per iteration
2. `24` MCTS simulations
3. `20` iterations
4. `120` heuristic eval games every `5` iterations
5. `40` head-to-head games against the current best model every eval cycle

Best heuristic eval result:

1. Iteration `20`
2. Winrate `40.8%`
3. Record `49-65-6`

Evaluation history file:
`checkpoints_mainline_c_v2/eval_history.jsonl`

Key evaluation trend:

1. Iteration `5`: `36.7%`
2. Iteration `10`: `38.3%`
3. Iteration `15`: `38.3%`
4. Iteration `20`: `40.8%`

Head-to-head against the current best model:

1. Iteration `10`: `50.0%`
2. Iteration `15`: `47.5%`
3. Iteration `20`: `45.0%`

Interpretation:

1. The mainline is improving against the heuristic benchmark.
2. Best-checkpoint selection is still noisy enough that head-to-head should remain part of evaluation.
3. Further progress is now more likely to come from AI/training optimization than from more rule changes.

## GPU Benchmark Status

Benchmark file:
`cli/benchmark_mainline_c_v2.json`

This benchmark was run around the current mainline rules and budget.

Best throughput candidates on the RTX 4070 Ti SUPER:

1. `parallel=10`, `infer_batch=40`, `train_batch=512` -> `12.23 pos/s`
2. `parallel=12`, `infer_batch=24`, `train_batch=384` -> `12.18 pos/s`
3. `parallel=10`, `infer_batch=40`, `train_batch=384` -> `11.98 pos/s`

Current recommended main training configuration:

1. `selfplay_parallel_games = 10`
2. `inference_batch_size = 40`
3. `batch_size = 512`

These settings have already been applied to `run_train_4070tis.ps1`.

## Files to Use on Another Device

The most important files to carry forward are:

1. `cli/tetris_weiqi.py`
2. `cli/analyze_rules.py`
3. `cli/train_alphazero.py`
4. `run_train_4070tis.ps1`
5. `run_rule_matrix_4070tis.ps1`
6. `TESTING_STATUS.md`

If you want the latest trained model as well, also copy:

1. `checkpoints_mainline_c_v2/best.pt`
2. `checkpoints_mainline_c_v2/eval_history.jsonl`
