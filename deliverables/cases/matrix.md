# Sleep Mode Test Cases

| ID | Path | Expected |
|----|------|----------|
| F1 | lightweight + inplace sleep/wake | success, no illegal FSM transition |
| F2 | destroy + recapture | success (legacy) |
| F3 | workspace_only | success |
| F4 | multi-round combined | accuracy/service stable |
| Q1 | UT state machine / rollback | all pass |
| N1 | wake while idle | idempotent |
