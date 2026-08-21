# Phase J / P0 Final Progress

## Attestation
- Branch: `plan/lightweight-sfc-refactor`
- Source commit S: `749003e099aeca43f0dd44ca5c1365765d648665`
- Evidence commit E: pending
- Formal Preliminary 300k: **NOT STARTED**
- Training command was not started.

## Phase J implementation status
- PPOTrainer and evaluation use `DecisionContext` and `ActionSubmission`; stale submissions are retried without buffer insertion, reward accounting, or decision-step consumption.
- Runtime reward is the frozen five-term J: alpha/uncovered=5, beta/distance=1, gamma/load_gap=1, delta/switches=0.25, eta/recovery_delay=0.5. Constraint violation is diagnostic only and excluded from J.
- Validation selection consumes authoritative `EpisodeMetrics`; missing metrics are hard failures. Fixed J is recomputed from the five frozen components.
- Formal protocol requires exactly 3 variants × 3 training seeds × 12 checkpoints at 25k increments through 300k; all run PPO hyperparameters are copied exactly except seed.
- Selection is per `(variant, training_seed)`, then freeze requires exactly 9 unique selections.
- Formal Test generation requires the nine-checkpoint freeze and the 200-tape Test bank; the ledger key includes variant, seed, checkpoint SHA, and Test manifest SHA and rejects formal repeat consumption.
- Formal resume is explicitly unsupported. Developer dry-run uses a separate namespace and `formal=false`.
- Direct CLI is available through `python run_phase_j.py` with `preliminary-train`, `preliminary-validate`, `preliminary-freeze`, `preliminary-test`, and `phase-j-dry-run`.

## Verification
- Locked environment: Python 3.11.5; torch 2.5.0+cpu; numpy 2.0.2; sb3-contrib 2.9.0; stable-baselines3 2.9.0; gymnasium 1.3.0.
- Full required suite: **94/94 PASS**.
- Fresh-directory dry-run: **PASS**, `phase_j_dry_run_749003e/phase_j_dry_run_summary.json`; 18 checkpoints, 9 selections, 9 freezes, `official_test_namespace_touched=false`.
- Final smoke: **PASS**, 80 tapes = 20 Single + 20 Sequential + 20 Overlap + 20 Burst; metadata commit equals S.
- Gate generation was performed only with the protected source/config/test tree clean and hashes are bound to Git blobs at S.
- P0 gate: `training_allowed=true`, `violations=[]`, top-level test count 94.

## Final hashes
- source_tree_hash: `6dc45e0718a855e37373ee0467dd27dfaf8fc268415a7728408f173311763712`
- protocol_sha256: `ac271cfff9fbfbaf03bba9143fa4a23ca03f36fe00fb7d5374f481d478188039`
- seed_manifest_sha256: `4fa2ab0ef3615b41d98ce4b1eca5072416ef326aa105a339ab23939a604ca615`
- smoke_summary_sha256: `4e761ea1de18da240bfc246180fd6f483de0e4275f9c5a6a963bfd34ba298960`
- smoke_manifest_sha256: `cac128e933df18e898f26c4a7f7d387bcb65e0e4d5479c26cd75efbea93f66b2`
- smoke_environment_metadata_sha256: `7f56d8711526bdccc01e819b6867d660f366e85a59a2281c720f1f3de3a43a16`

## Remaining limitations
- No formal Preliminary result exists; 300k training remains intentionally unstarted pending independent audit.
- Formal Validation/Freeze/Test artifacts will only be created after formal training and independent approval.
- L2 SFC/Isaac Sim integration has not started.
