$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

$commonArgs = @(
  'cli\train_alphazero.py',
  '--device', 'cuda',
  '--amp',
  '--dead-zone-fills-line', 'true',
  '--score-dead-zone-weight', '0.0',
  '--piece-distribution', 'bag7',
  '--terminal-mode', 'pieces_only',
  '--allow-voluntary-skip', 'false',
  '--games-per-iter', '24',
  '--selfplay-parallel-games', '8',
  '--num-simulations', '24',
  '--inference-batch-size', '32',
  '--batch-size', '256',
  '--iterations', '20',
  '--min-train-batches', '16',
  '--eval-every', '5',
  '--eval-games', '120',
  '--eval-num-simulations', '12',
  '--eval-parallel-games', '8'
)

$variants = @(
  @{
    Name = 'A_baseline'
    SaveDir = 'checkpoints_rule_A_baseline'
    EndCondition = 'double_forced_pass'
    NoLegalMove = 'pass_and_redraw'
  },
  @{
    Name = 'B_strict_end'
    SaveDir = 'checkpoints_rule_B_strict_end'
    EndCondition = 'single_forced_pass'
    NoLegalMove = 'pass_and_redraw'
  },
  @{
    Name = 'C_reroll_buffer'
    SaveDir = 'checkpoints_rule_C_reroll_buffer'
    EndCondition = 'double_forced_pass'
    NoLegalMove = 'reroll_once_then_pass'
  }
)

foreach ($variant in $variants) {
  Write-Host ''
  Write-Host ('=' * 72)
  Write-Host ("[Train] {0}" -f $variant.Name)
  Write-Host ("  save_dir={0}" -f $variant.SaveDir)
  Write-Host ("  end_condition_mode={0}" -f $variant.EndCondition)
  Write-Host ("  no_legal_move_mode={0}" -f $variant.NoLegalMove)
  Write-Host ('=' * 72)

  & python @commonArgs `
    --save-dir $variant.SaveDir `
    --end-condition-mode $variant.EndCondition `
    --no-legal-move-mode $variant.NoLegalMove

  if ($LASTEXITCODE -ne 0) {
    throw "Training failed for $($variant.Name)"
  }
}

