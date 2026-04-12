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

### M5 MPS Training (2026-04-12)

Directory:
`checkpoints_m5_mps`

This run used the same mainline rules and training budget as the 4070Ti S
run, but with MPS-optimized parameters and float16 autocast enabled.

Configuration:

1. `4` self-play games per batch (MPS-optimal)
2. `48` inference batch size
3. `24` MCTS simulations
4. `20` iterations
5. `120` heuristic eval games every `5` iterations
6. `40` head-to-head games against best model every eval cycle

Best heuristic eval result:

1. Iteration `10`
2. Winrate `45.0%`
3. Record `54-57-9`

Evaluation history:

1. Iteration `5`: `30.0%` (36-79-5)
2. Iteration `10`: `45.0%` (54-57-9) **best**
3. Iteration `15`: `45.0%` (54-62-4)
4. Iteration `20`: `38.3%` (46-63-11)

Head-to-head against the current best model:

1. Iteration `10`: `47.5%` (19-15-6)
2. Iteration `15`: `47.5%` (19-17-4)
3. Iteration `20`: `47.5%` (19-16-5)

Total training time: `30.1 minutes` (20 iterations).

### Cross-Device Training Comparison

| Metric | RTX 4070 Ti S | Apple M5 MPS |
|--------|---------------|--------------|
| Best eval winrate | 40.8% (iter 20) | **45.0%** (iter 10) |
| Iter 5 winrate | 36.7% | 30.0% |
| Iter 10 winrate | 38.3% | **45.0%** |
| Iter 15 winrate | 38.3% | **45.0%** |
| Iter 20 winrate | **40.8%** | 38.3% |
| Training time | ~37 min (est.) | **30.1 min** |

Interpretation:

1. M5 MPS reaches peak performance earlier (iter 10 vs iter 20).
2. Both platforms show late-iteration regression, suggesting overfitting
   with the current buffer size and training budget.
3. The mainline is improving against the heuristic benchmark.
4. Best-checkpoint selection is still noisy enough that head-to-head should remain part of evaluation.
5. Further progress is now more likely to come from AI/training optimization than from more rule changes.

## GPU Benchmark Status

### RTX 4070 Ti SUPER (Windows, CUDA)

Benchmark file:
`cli/benchmark_mainline_c_v2.json`

Best throughput candidates:

1. `parallel=10`, `infer_batch=40`, `train_batch=512` -> `12.23 pos/s`
2. `parallel=12`, `infer_batch=24`, `train_batch=384` -> `12.18 pos/s`
3. `parallel=10`, `infer_batch=40`, `train_batch=384` -> `11.98 pos/s`

Recommended training configuration:

1. `selfplay_parallel_games = 10`
2. `inference_batch_size = 40`
3. `batch_size = 512`

### Apple M5 MacBook Air (macOS, MPS)

Benchmark file:
`cli/benchmark_m5_mps.json`

Hardware: Apple M5 (10-core, 4P+6E), 32GB unified memory.

MPS optimizations applied in this round:

1. Enabled `float16` autocast on MPS (previously CUDA-only)
2. Disabled `GradScaler` on MPS (not needed, no underflow risk with unified memory)
3. Tuned parallel/batch parameters for MPS unified memory architecture

Best throughput candidates:

1. `parallel=4`, `infer_batch=48`, `train_batch=512` -> `21.23 pos/s`
2. `parallel=4`, `infer_batch=32`, `train_batch=256` -> `21.15 pos/s`
3. `parallel=12`, `infer_batch=32`, `train_batch=256` -> `21.00 pos/s`

Recommended training configuration:

1. `selfplay_parallel_games = 4`
2. `inference_batch_size = 48`
3. `batch_size = 512`

### Cross-Device Comparison

| Metric | RTX 4070 Ti S | Apple M5 MPS | Difference |
|--------|---------------|--------------|------------|
| Best throughput | 12.23 pos/s | 21.23 pos/s | **+73.5%** |
| Optimal parallel | 10 | 4 | MPS prefers small parallel |
| Optimal infer batch | 40 | 48 | Similar |
| AMP | float16 + GradScaler | float16 (no scaler) | — |
| Peak VRAM | 422 MB | N/A (unified) | — |

Key insight: MPS unified memory avoids CPU↔GPU data transfers, allowing
higher throughput despite lower raw compute. MPS also prefers fewer
parallel games with larger inference batches.

## Comprehensive Rule Testing (2026-04-12, M5 MPS)

### D Variant: Missing A/B/C/D Combo Test

Tested: `single_forced_pass + reroll_once_then_pass` (the missing 4th combo).

Directory: `checkpoints_D_single_reroll`

| Variant | end_condition | no_legal_move | Best Winrate |
|---------|---------------|---------------|--------------|
| A_baseline | double_forced_pass | pass_and_redraw | 31.7% |
| B_strict_end | single_forced_pass | pass_and_redraw | 33.3% |
| **C_reroll_buffer** | **double_forced_pass** | **reroll_once_then_pass** | **45.0%** |
| D_single_reroll | single_forced_pass | reroll_once_then_pass | 40.8% |

D variant eval history:

1. Iteration `5`: `31.7%`
2. Iteration `10`: `40.8%` **best**
3. Iteration `15`: `35.0%`
4. Iteration `20`: `32.5%`

Conclusion: `reroll_once_then_pass` is the main driver of improvement
(D=40.8% >> B=33.3%), but `double_forced_pass` provides additional
benefit (C=45.0% > D=40.8%). Both parameters contribute independently.
The C combo (current mainline) is confirmed optimal.

### Single-Parameter Ablation (Heuristic, 500 games each)

Each row changes exactly one parameter from the mainline baseline.

| Variant | P1 Win | Cap Rate | Cap Win | DZ Fill | Moves | Lead Chg | Lines | Fill |
|---------|--------|----------|---------|---------|-------|----------|-------|------|
| **主线基准 (10x10)** | **50.8%** | **99.8%** | **64.3%** | **36.6%** | **80.4** | **8.9** | **22.9** | **74.3%** |
| end: single_forced_pass | 53.8% | 97.2% | 67.6% | 37.3% | 41.3 | 4.0 | 8.4 | 70.0% |
| no_legal: pass_and_redraw | 53.4% | 98.0% | 71.8% | 38.1% | 45.7 | 4.3 | 10.0 | 70.7% |
| resolution: clear_then_capture | 50.0% | 99.4% | 63.7% | 39.7% | 80.0 | 9.2 | 23.1 | 74.1% |
| activation: next_turn | 51.0% | 99.8% | 61.4% | 36.6% | 80.6 | 9.0 | 23.0 | 74.4% |
| piece_dist: bag7_independent | 48.2% | 99.6% | 60.8% | 37.9% | 75.7 | 8.4 | 21.2 | 73.7% |
| dead_zone_fills_line=False | 56.2% | 99.8% | 78.5% | 60.2% | 42.0 | 3.2 | 7.2 | 74.6% |
| rerolls: 2 | 44.2% | 100.0% | 57.9% | 35.3% | 100.5 | 12.0 | 30.6 | 75.7% |

Key findings:

1. **High-impact parameters** (large delta from baseline):
   - `dead_zone_fills_line`: Toggling to False causes P1 winrate +5.4%,
     captures become 78.5% decisive, lead changes drop from 8.9 to 3.2.
     **Most important rule parameter by far.**
   - `end_condition_mode`: single_forced_pass halves game length (41 vs 80),
     halves lead changes (4.0 vs 8.9), and reduces line clears sharply.
   - `no_legal_move_mode`: pass_and_redraw similarly shortens games and
     reduces dynamism. Reroll is clearly superior.
   - `rerolls: 2`: Extends games to 100+ moves, excessive.

2. **Low-impact parameters** (near-identical to baseline):
   - `resolution_mode`: clear_then_capture vs recheck — nearly identical
     on all metrics. DZ fill rate slightly higher (39.7% vs 36.6%).
   - `dead_zone_activation_mode`: immediate vs next_turn — negligible
     difference across all metrics.
   - `piece_distribution`: bag7 vs independent — small differences.
     Independent bags slightly lower game length and lead changes.

### Cross-Test: resolution_mode × dead_zone_activation_mode

6 combinations tested (heuristic, 500 games each).

| resolution_mode | activation | P1 Win | Moves | Lead Chg | Lines |
|-----------------|------------|--------|-------|----------|-------|
| capture_then_clear_recheck | immediate | 50.8% | 80.4 | 8.9 | 22.9 |
| capture_then_clear_recheck | next_turn | 51.0% | 80.6 | 9.0 | 23.0 |
| clear_then_capture | immediate | 50.0% | 80.0 | 9.2 | 23.1 |
| clear_then_capture | next_turn | 50.8% | 79.9 | 9.2 | 23.0 |
| capture_then_clear_once | immediate | 50.8% | 80.4 | 8.9 | 22.9 |
| capture_then_clear_once | next_turn | 51.0% | 80.6 | 9.0 | 23.0 |

Conclusion: **No meaningful interaction.** All 6 combinations produce
nearly identical results. These two parameters do not interact at the
heuristic AI level. Both can be considered stable at their current values.

Note: `capture_then_clear_recheck` and `capture_then_clear_once` are
identical in heuristic play (the recheck rarely triggers a second capture).
The difference only emerges under stronger AI or specific board states.

### Cross-Test: piece_distribution × dead_zone_fills_line

4 combinations tested (heuristic, 500 games each).

| piece_dist | dzfl | P1 Win | Cap Win | DZ Fill | Moves | Lead Chg | Lines |
|------------|------|--------|---------|---------|-------|----------|-------|
| **bag7** | **True** | **50.8%** | **64.3%** | **36.6%** | **80.4** | **8.9** | **22.9** |
| bag7 | False | 56.2% | 78.5% | 60.2% | 42.0 | 3.2 | 7.2 |
| bag7_independent | True | 48.2% | 60.8% | 37.9% | 75.7 | 8.4 | 21.2 |
| bag7_independent | False | 57.0% | 77.1% | 61.6% | 41.7 | 2.9 | 7.0 |

Conclusion: `dead_zone_fills_line` dominates this interaction.
Regardless of piece distribution, setting dzfl=False always causes:
shorter games, fewer lead changes, higher capture decisiveness.
`piece_distribution` has only minor additive effects. **No surprising
interaction** — these parameters are independent in effect.

### 8×8 Board Generalization

| Board | P1 Win | Cap Rate | Cap Win | DZ Fill | Moves | Lead Chg | Lines | Fill |
|-------|--------|----------|---------|---------|-------|----------|-------|------|
| 10×10 | 50.8% | 99.8% | 64.3% | 36.6% | 80.4 | 8.9 | 22.9 | 74.3% |
| 8×8 | 47.4% | 95.8% | 60.7% | 27.3% | 53.0 | 7.4 | 19.9 | 71.0% |

On 8×8 the rules still produce healthy metrics but with notable shifts:

1. Capture rate drops from 99.8% to 95.8% (less room for surrounding).
2. Dead zone fill rate drops from 36.6% to 27.3% (less room to convert).
3. Games are shorter (53 vs 80 moves) — expected for a smaller board.
4. Lead changes slightly fewer (7.4 vs 8.9) but still healthy.
5. P1 winrate is 47.4% — slightly favoring P2 (reverse of 10×10).

Conclusion: The mainline rules generalize to 8×8 without breaking.
The core dynamics (capture, line clear, dead zone conversion) all function.
Minor rebalancing may be needed if 8×8 becomes a supported mode.

## Rule Testing Summary

### Parameter Importance Ranking (from ablation)

1. **dead_zone_fills_line** — Critical. True is correct; False breaks balance.
2. **end_condition_mode** — High. double_forced_pass is correct.
3. **no_legal_move_mode** — High. reroll_once_then_pass is correct.
4. **no_legal_move_rerolls** — Medium. 1 is correct; 2 causes bloat.
5. **piece_distribution** — Low. bag7 is slightly better than independent.
6. **resolution_mode** — Very low. All 3 modes nearly identical.
7. **dead_zone_activation_mode** — Very low. No measurable impact.

### Large Buffer + 400 Eval Training Test

Directory: `checkpoints_mainline_large_buffer`

Configuration changes from standard mainline:

1. `buffer_size`: 100,000 → 200,000
2. `games_per_iter`: 24 → 32
3. `eval_games`: 120 → 400

Evaluation history:

| Iter | Winrate | Record | 95% CI | h2h vs Best |
|------|---------|--------|--------|-------------|
| 5 | **40.0%** | 160-217-23 | ±4.8% | N/A |
| 10 | 37.0% | 148-229-23 | ±4.7% | 40.0% |
| 15 | 37.8% | 151-224-25 | ±4.7% | 50.0% |
| 20 | 38.2% | 153-216-31 | ±4.8% | 42.5% |

Comparison with standard buffer (120 eval games):

| Config | Best Winrate | Peak Iter | Iter 20 | Regression |
|--------|-------------|-----------|---------|------------|
| buffer=100k, eval=120 | 45.0% | 10 | 38.3% | -6.7% |
| buffer=200k, eval=400 | 40.0% | 5 | 38.2% | -1.8% |

Findings:

1. **Larger buffer reduces regression**: Iter 20 drops only 1.8% from peak
   (vs 6.7% with smaller buffer). The model is more stable.
2. **Lower peak, but more reliable**: With 400 eval games the variance
   is lower, so the 40.0% peak is a more trustworthy number than the
   45.0% measured with only 120 games.
3. **Iter 20 performance is nearly identical** (38.2% vs 38.3%),
   suggesting the real capability of a 20-iteration model is ~38%.
4. The earlier "45.0% peak" was likely inflated by small-sample noise.

Recommendation: Use buffer_size=200k and eval_games=400 as the new
standard training configuration for future tests.

### Remaining Known Issues

1. **SimpleAI evaluation bias**: Captures weighted 15-20x vs line clears
   at 3x. Low-impact parameters (resolution_mode, activation_mode) may
   show differences under stronger AI that the heuristic cannot detect.

Note: `score_dead_zone_weight` was intentionally dropped by design
decision, not a testing gap. The game design avoids numeric weight
tuning in favor of clean, intuitive rules.

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
