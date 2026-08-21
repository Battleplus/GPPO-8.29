# Phase J / P0 Final Progress

## Final commits (S3 round)
- Branch: `plan/lightweight-sfc-refactor`
- Source S3: `f87fd5acc3cc458127099ccca90f3d09a092a4a9`
- Evidence E3: pending until this evidence commit is created
- Formal Preliminary 300k: **NOT STARTED**

## S3 blocker fixes (final pre-300k closeout)
- Frozen train mode cycle: `preliminary_train` now loads
  `seed_manifest.preliminary.train.mode_cycle` (sequential/overlap/burst) and
  passes it to `CyclingTrainingEnv`; the generic evaluation `MODES` constant
  is no longer used for training.
- Train seed namespace: `episodes_per_training_seed` expanded 1000 -> 300000
  so the frozen 300k decision budget fits inside the reserved namespace even
  under the worst-case 1 accepted decision per episode. Seed ranges for
  1101/2202/3303 and the runtime formula
  `training_seed*1000003+episode_index` /
  `training_seed*10000019+episode_index` are unchanged and machine-checked.
- `CyclingTrainingEnv.reset()` hard-FAILs once `reset_index >= max_resets`
  (frozen reserved cap) instead of silently leaving the train namespace.
- Per-event `fixed_j` is materialised in `EventMetrics` at episode
  finalisation. An event that never produced a decision (unobserved, or an
  earlier event caused final infeasible termination) uses the final
  environment cost snapshot plus the frozen 200s recovery horizon; `None` is
  never silently treated as zero. `extract_validation_metrics` verifies and
  sums the materialised values, so multi-event early termination stays
  lexicographically rankable.
- Formal Test partial resume: consumed ledger entries are skipped only when
  their checkpoint/test/freeze/source/protocol/seed provenance matches the
  current lock; every unconsumed checkpoint is written as state=running
  before evaluation and flipped to state=consumed only after the result file
  is durable. A completed lock or a leftover running state hard-fails.

## Verification on S3
- Locked environment: Python 3.11.5; torch 2.5.0+cpu; numpy 2.0.2;
  sb3-contrib 2.9.0; stable-baselines3 2.9.0; gymnasium 1.3.0.
- Full suite: **110/110 PASS** (8 new tests added this round).
- Fresh developer dry-run: **PASS**, 18 checkpoints -> 9 selected -> 9 frozen;
  official Test namespace untouched.
- Final smoke: **PASS**, 20 Single + 20 Sequential + 20 Overlap + 20 Burst =
  80; metadata git_commit equals S3.
- P0 Gate: `training_allowed=true`, `violations=[]`, top-level
  `test_count=110`, attested_source_commit_sha == S3.
- `_check_p0_gate()` on the evidence descendant: PASS.

## S3 Gate hashes
- attested/source commit: `f87fd5acc3cc458127099ccca90f3d09a092a4a9`
- seed_namespace_isolation check: train_mode_cycle_ok, runtime_formula_ok,
  seed_coverage_ok, seed_start_formula_ok all true; train/validation/test
  namespaces disjoint.
- smoke metadata git_commit == `f87fd5acc3cc458127099ccca90f3d09a092a4a9`

## Remaining
- Formal Preliminary training/Validation/Freeze/Test have not been run.
- No 300k training was started.
- L2 SFC/Isaac Sim integration has not started.

## S4 closeout round (final audit items)
- **Train reset cap off-by-one**: `episodes_per_training_seed` 300000 -> 300001.
  Worst case is 1 initial reset + 300000 post-terminal resets (each accepted
  terminal transition triggers a reset, including the final requested step), so
  `reserved >= formal_budget + 1` is now required and machine-checked by the
  gate (`seed_coverage_rule`, `seed_coverage_assertions_ok`). Trainer-level test
  `test_trainer_N_accepted_steps_requires_N_plus_1_resets` proves N accepted
  steps need exactly N+1 reset calls (not only a `CyclingTrainingEnv.reset` test).
- **Formal Test crash atomicity**: per-checkpoint atomic journal
  (same-dir temp + fsync + os.replace via `_json_file`). Journal is the single
  source of truth. States: A (running + no ledger) hard-fails; B (consumed
  journal + ledger missing) rebuilds the ledger only after cryptographically
  verifying the persisted result, never re-evaluates; C (ledger consumed +
  journal running) restores the journal from the verified ledger/result,
  never re-evaluates; D (both consumed + exact provenance) skips; E (9/9
  completed lock) rejects any further run. Crash-state tests assert
  `run_episode` is never called for B/C.
- **metrics.py protected**: `ppo_allocation/random_event/metrics.py` added to
  P0 `SOURCE_FILES`; contract test asserts environment.py, experiment.py,
  metrics.py, models.py, trainer.py, reward.py, phase_j.py are all protected.
- **Real early-termination integration test**: a 5-event burst tape (4 UAV_DAMAGE
  + 1 later REGION_VACANCY) is replayed by `run_episode`; the burst terminates
  the episode with final infeasibility before the REGION_VACANCY is observed;
  that event has decision_count=0, recovery_delay=None and a finite fixed_j
  produced by run_episode itself (no pre-filled values); the episode stays
  lexicographically rankable.
- **Frozen unobserved fixed-J rule**: `fixed_j_unobserved_event_rule` written
  into `configs/random_event_protocol.json` (final-snapshot uncovered/distance/
  load_gap, frozen switches=0, 200s censored recovery). Runtime uses the shared
  `UNOBSERVED_EVENT_RECOVERY_PENALTY_SECONDS` constant in reward.py; the gate
  machine-compares rule vs runtime constant and probes
  `compute_cost(env).switches == 0.0`.

## Verification on S4
- Locked Python 3.11.5 + torch 2.5.0+cpu + numpy 2.0.2 + sb3-contrib 2.9.0 +
  stable-baselines3 2.9.0 + gymnasium 1.3.0.
- Full suite: **117/117 PASS** (7 new tests this round).
- Fresh developer dry-run on S4: PASS, 18 checkpoints -> 9 selected -> 9 frozen,
  official Test namespace untouched.
- Final 20x4 smoke regenerated on S4: 80 tapes replayed; metadata git_commit ==
  S4 `0b48d20c58028503818a953ced93216912db1c4a`.
- P0 Gate GREEN: `training_allowed=true`, `violations=[]`, `test_count=117`,
  `attested_source_commit_sha == 0b48d20c...`.
- `_check_p0_gate()` on the E4 evidence descendant: PASS.
- `handoff/CURRENT_STATE.json` no longer stores the evidence commit's own SHA
  (self-reference); it records the exact attested source SHA and points to
  current branch HEAD for the evidence commit.

## Remaining
- Formal Preliminary training/Validation/Freeze/Test have not been run.
- No 300k training was started.
- L2 SFC/Isaac Sim integration has not started.
