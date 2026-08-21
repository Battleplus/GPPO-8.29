# Phase J / P0 Final Progress

## Final commits
- Branch: `plan/lightweight-sfc-refactor`
- Source S2: `6943a92f7e3663fc469eea98ad730e64784a0148`
- Evidence E2: pending until this evidence commit is created
- Formal Preliminary 300k: **NOT STARTED**

## S2 blocker fixes
- Unrecovered Validation events remain rankable. Level 1 always uses final infeasible count/rate. Level 3 uses observed recovery-delay sum plus a frozen 200.0-second censoring penalty per unresolved event. Level 4 uses the same explicit penalty in frozen J; `None` is never silently treated as zero.
- Formal freeze requires `selection.formal=true`, reruns the formal Validation manifest contract, and matches source-tree, attested-commit, protocol, seed-manifest, and Validation manifest SHA.
- Formal Test requires a formal freeze with exactly the nine variant/seed keys and matching aggregate/entry provenance.
- Official Test generation is protected by `formal_test_bank_lock.json`; existing locks cannot regenerate the manifest. The lock includes Test manifest SHA, freeze manifest SHA, source tree, attested source commit, protocol, and seed-manifest hashes.
- Formal Test accepts partial resume only for unconsumed ledger entries against the same lock. A completed lock is terminal.
- The public `protocol-bank --tier preliminary --split test` command routes through the guarded Phase J Test generator and rejects before freeze.
- Trainer/evaluation tests inject one stale submission, verify re-inference, and verify no stale transition/reward/event metric is recorded.

## Verification on S2
- Locked environment: Python 3.11.5; torch 2.5.0+cpu; numpy 2.0.2; sb3-contrib 2.9.0; stable-baselines3 2.9.0; gymnasium 1.3.0.
- Full suite: **102/102 PASS**.
- Fresh developer dry-run: **PASS**, 18 checkpoints -> 9 selected -> 9 frozen; official Test namespace untouched.
- Final smoke: **PASS**, 20 Single + 20 Sequential + 20 Overlap + 20 Burst = 80; metadata commit equals S2.
- P0 Gate: `training_allowed=true`, `violations=[]`, top-level `test_count=102`.
- `_check_p0_gate()` on the evidence descendant: PASS.

## Hashes
- attested/source commit: `6943a92f7e3663fc469eea98ad730e64784a0148`
- source_tree_hash: `d6370e4aab14a496b89914298f4b48ccf1987d81a87db2af4254f5e51d342a9e`
- protocol_sha256: `ad454162cf327af130954f91f23fb53a566346843b5516b5b7a6d74d3ae7879f`
- seed_manifest_sha256: `4fa2ab0ef3615b41d98ce4b1eca5072416ef326aa105a339ab23939a604ca615`
- smoke_summary_sha256: `55507a932cb0c0802ddd57c539aed10cfb6276a9610eed704760d4d8d3ce63de`
- smoke_manifest_sha256: `58668ae1a88bcfb5c64568c718a71986c2df3ad83f1f90663b18fd6acbebd421`
- smoke_environment_metadata_sha256: `7d010b9d34d90841ae1bdfd1c0fa45d9f8ac7eb80bf08cb21347ddf19f20f54f`

## Remaining
- Formal Preliminary training/Validation/Freeze/Test have not been run.
- No 300k training was started.
- L2 SFC/Isaac Sim integration has not started.
