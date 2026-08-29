# Extreme multi-event stress test

This directory contains a post-hoc exploratory stress test for the fixed 50k
PPO-MLP and GPPO-Adaptive checkpoints.  It is intentionally isolated from the
formal minimum-validation contract and must not be interpreted as a second
held-out test or as checkpoint-selection evidence.

## Provenance and scope

- Runtime base: `f14aa207823631bdda6a38a5663190341b7e4a3e`
- Green evidence head: `2afa8ec1cb481deb57645dbd30240d90d32d2233`
- Attested source: `32974ec85be71e192b12cae85d00eb877d5fe07d`
- Checkpoint step: fixed `50,000`, no checkpoint selection
- Frozen policies: PPO-MLP and GPPO-Adaptive, seeds `1101`, `2202`, `3303`
- Training performed by this experiment: none
- Formal Validation/Freeze/Test modified by this experiment: none

The seven stress families cover atomic multi-event shocks, three-UAV resource
collapse, tracking saturation with delayed release, causally out-of-order
reports, a long blind burst, rapid task churn, and an eight-event storm.

## Results

- [Chinese research conclusion](CONCLUSION_ZH.md)
- [Interpretation](results_20260827/INTERPRETATION.md)
- [Full scenario report](results_20260827/REPORT.md)
- [Run summary](results_20260827/run_summary.json)
- [Aggregate metrics](results_20260827/aggregate_results.csv)
- [Paired GPPO-minus-PPO effects](results_20260827/paired_effects_gppo_minus_ppo.csv)
- [In-flight stale-decision audit](results_20260827/stale_decision_race_audit.json)
- [SHA-256 inventory](results_20260827/sha256_inventory.json)
- [Chinese briefing for the senior researcher](SENIOR_BRIEFING_ZH.md)

The archive includes all 42 immutable tapes and all 420 raw episode traces.
Every trace satisfies the reward invariant; all episodes terminated normally,
with no invalid or repaired actions.

## Reproduction

The Git branch intentionally does not contain model checkpoint binaries.  A
local `ppo_allocation` artifact directory containing the frozen manifest and
the six verified 50k checkpoints is required:

```powershell
python scripts\run_extreme_scenarios.py `
  --checkpoint-root E:\path\to\ppo_allocation `
  --output-dir E:\new\empty\extreme_results
```

The runner refuses to reuse a non-empty output directory and verifies all six
checkpoint SHA-256 values before replaying any episode.

To audit an event arriving after inference but before action submission:

```powershell
python scripts\audit_stale_decision_race.py `
  --checkpoint-root E:\path\to\ppo_allocation `
  --result-root E:\path\to\extreme_results
```

## Interpretation boundary

The bank was designed after reviewing the original held-out results, so it is
development evidence.  Any new algorithm developed against these cases must
be evaluated once on a newly frozen, unseen Extreme-V2 bank before making a
comparative claim.
