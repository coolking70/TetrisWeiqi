$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

python cli\train_alphazero.py `
  --device cuda `
  --amp `
  --dead-zone-fills-line true `
  --score-dead-zone-weight 0.0 `
  --piece-distribution bag7 `
  --terminal-mode pieces_only `
  --allow-voluntary-skip false `
  --end-condition-mode double_forced_pass `
  --no-legal-move-mode reroll_once_then_pass `
  --resolution-mode capture_then_clear_once `
  --dead-zone-activation-mode immediate `
  --no-legal-move-rerolls 1 `
  --games-per-iter 24 `
  --selfplay-parallel-games 10 `
  --num-simulations 16 `
  --inference-batch-size 40 `
  --batch-size 512 `
  --lr 0.0015 `
  --lr-step-size 15 `
  --lr-gamma 0.9 `
  --dirichlet-alpha 0.05 `
  --dirichlet-epsilon 0.10 `
  --iterations 20 `
  --train-steps-per-iter 16 `
  --eval-every 5 `
  --eval-games 120 `
  --eval-num-simulations 12 `
  --eval-parallel-games 8 `
  --eval-headtohead-games 40 `
  --eval-seed-base 1000 `
  --eval-report-name eval_history.jsonl `
  --save-dir checkpoints_mainline_c_v2
