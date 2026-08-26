param(
  [string]$Root = "",
  [int]$RefreshSeconds = 5
)
$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($Root)) {
  $Root = Join-Path (Get-Location) "results\random_event\minimum_validation_50k_run1\preliminary\runs"
}
$variants = @("PPO-MLP", "GPPO-Adaptive")
$seeds = @(1101,2202,3303)
while ($true) {
  Clear-Host
  $total = 0
  $done = 0
  $checkpoints = 0
  $rows = @()
  foreach ($seed in $seeds) {
    foreach ($variant in $variants) {
      $path = Join-Path $Root "$variant\seed_$seed\progress\live_progress.json"
      if (Test-Path -LiteralPath $path) {
        try { $p = Get-Content -LiteralPath $path -Raw | ConvertFrom-Json } catch { $p = $null }
      } else { $p = $null }
      $steps = if ($null -ne $p) { [int]$p.total_steps } else { 0 }
      $status = if ($null -ne $p) { [string]$p.status } else { "pending" }
      $rate = if ($null -ne $p) { [double]$p.steps_per_second } else { 0 }
      $total += $steps
      if ($status -eq "done") { $done++ }
      if ($null -ne $p) { $checkpoints += [int]$p.checkpoint_count }
      $rows += [pscustomobject]@{ Variant=$variant; Seed=$seed; Steps="$steps/50000"; Percent=(100*$steps/50000); Status=$status; StepsPerSecond=$rate }
    }
  }
  Write-Host ("Campaign: {0:P1}" -f ($total/300000))
  Write-Host "Workers: 2 per seed batch | Completed runs: $done/6 | Checkpoints: $checkpoints/12"
  Write-Host ("Aggregate steps/s: {0:N2}" -f (($rows | Measure-Object StepsPerSecond -Sum).Sum))
  $rows | Format-Table -AutoSize
  Start-Sleep -Seconds $RefreshSeconds
}
