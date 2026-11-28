# Command test entries

Load a JSON file once, before any checkout. The command and environment stay
fixed across rounds; they are not inferred from YAML names or cluster discovery.

```json
{
  "command": ["python", "-u", "-m", "pytest", "tests/example.py"],
  "working_directory": ".",
  "environment": {},
  "timeout_seconds": 1800
}
```

Example:

```bash
python -m tools.bisect.auto_bisect \
  --scene single_node --soc a3 \
  --good-commit GOOD --bad-commit BAD \
  --test-entry /outside-repository/test-entry.json
```

Exit 0 means PASS, 125 means SKIP; other exit codes mean FAIL. Failure to start
or timeout means SKIP. There is no implicit pytest-code or benchmark-JSON
interpretation. Wrap the original test when it needs a different verdict rule.
The foreground entry must wait for all its work and stop its services before
returning. No nightly process-name cleanup runs for command entries.

Relative working directories resolve against the target repository. Scripts
inside that repository follow the checked-out commit. Put a fixed reproduction
script outside the checkout when it must survive version changes. A definition
hash separates state/cache directories, but cannot detect edits to external
script contents or changes to inherited ambient variables; use a new work dir
when those inputs change.

For multiple nodes, retain the existing explicitly configured worker protocol:
start on every participating environment with the same shared coordination
folder, explicit node count and unique node index. Every node runs the same
entry; its script may use the existing per-node deployment environment. Ready
messages verify definition identities. Any node SKIP makes the round SKIP;
otherwise any node FAIL makes it FAIL. This does not discover Ray clusters,
Kubernetes Pods or execution targets. No transport is auto-selected.

The analyzer, planner and local deployment executor remain responsible for
preparing each node. No model test is run in mock mode, but preparation is real.
Legacy --config-yaml remains available as a mutually exclusive compatibility
entry. Command mode requires an explicit good commit and does not use the nightly
good table.

For controller-driven Ray or Kubernetes participants, see
[execution targets](../execution/README.md). Supplying `--execution-file`
replaces manual worker startup.

Command execution lives in `executor.py`, raw results in `outcome.py`, and
synthetic mock verdicts in `mock.py`. Nightly test paths, metadata and cleanup
live in `nightly.py`, with benchmark verdicts in `nightly_verdict.py`.
The deployment/test sequence is coordinated by
`../orchestration/nightly_trial.py`. Nightly good-commit lookup uses
`good_table.py`.
