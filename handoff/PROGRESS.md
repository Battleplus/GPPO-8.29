# P0 Gate Progress

## Current Status
- **training_allowed**: true
- **gate**: machine-generated, training_allowed=true, zero violations
- **source commit**: fcd14b25fc50b3c54be86f93a7ec2f1dcd7cd348
- **gate commit**: ddf7c8a1493b28f23a92a88dc128e93ace0c6971

## Phase Completion

| Phase | Status | Notes |
|-------|--------|-------|
| A: Trusted State | ✅ | gate P0, training_allowed=false initially |
| B: Frozen Protocol | ✅ | seeds [1101,2202,3303]; V=100; T=200 |
| C: Runtime Bridge | ✅ | TruthEvent→Detector→Observation→Confirmation→Belief |
| D: Confirmation | ✅ | 3-of-5 discovery, dual-path destruction, heartbeat/probe |
| E: Concurrency | ✅ | command/ACK/lease/fencing full lifecycle |
| F: Fair PPO-MLP | ✅ | input_dim=384, three variants save/load |
| G: Event Modes | ✅ | burst delta==1, four-mode reward invariant |
| H: Auto P0 Gate | ✅ | training_allowed=true, 71/71 tests ALL PASS |

## Round-4 Fixes (fcd14b2)
- DecisionContext dataclass: begin_decision returns graph_version + action_version
- submit_action validates both graph_version and action_version
- E2E stale action_version test via real env API
- Gate expanded: 5 graph-stale + 5 action-version-stale injection
- Smoke provenance: metadata.git_commit == attested_source_commit_sha
- Smoke manifest SHA256 (non-empty 64 hex)
- Fencing probe: old lease/ACK rejected, new token > old
- SOURCE_FILES expanded to 26 protected files
- stale_attempted removed from normal action path

## Test Results
- **71 tests, 0 failures, 0 errors** (Python 3.11.5, torch 2.5.0+cpu, numpy 2.0.2)

## Smoke Evidence
- **80 tapes** (20 Single, 20 Sequential, 20 Overlap, 20 Burst)
- Python 3.11.5, frozen packages verified
- git_commit matches attested source commit
- manifest SHA256: non-empty, recorded in gate

## Remaining
- Preliminary 300k training: gate允许, 未启动 (等待用户确认)
- L2 SFC/Isaac Sim: 未开始 (按规划 L0/L1 稳定后)
