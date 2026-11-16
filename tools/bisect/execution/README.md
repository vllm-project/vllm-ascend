# Existing execution targets

Supply `--test-entry test.json --execution-file targets.json` to drive existing
participants from one controller. No deployment or target discovery occurs.
Each round probes all targets, prepares all targets, then starts all tests.
A transport failure stops the controller: unknown completion cannot start a
new round. Mock mode performs real preparation and omits only test execution.

```json
{
  "prepare_timeout_seconds": 7200,
  "nodes": [{
    "name": "worker0",
    "channel": "kubernetes",
    "repository": "/vllm-workspace/vllm-ascend",
    "python": "/root/.cache/nightly_bisect/tool_venv/bin/python",
    "context": "test-cluster",
    "namespace": "default",
    "pod": "existing-worker",
    "container": "model",
    "pod_uid": "EXACT-POD-UID",
    "container_id": "containerd://EXACT-CONTAINER-ID"
  }]
}
```

Local nodes have only `name`, `channel: "local"`, `repository`, and `python`.
Ray nodes replace Kubernetes fields with `address` and `node_id`, and select
`channel: "ray"`. Mixed transports are supported, with one Ray address per
controller. Ray uses hard node affinity and disables retries. Kubernetes
checks Pod UID and container ID before and after exec.

The controller needs kubectl credentials or the optional Ray dependency.
Selected target Python interpreters need the analyzer requirements installed.
Probing never installs packages. Tool code is snapshotted outside the checkout
so switching commits cannot replace the running control implementation.

Node state and logs live under
`~/.cache/nightly_bisect/execution/<session>/<node>/`.
Controller logs also contain per-round, per-node probe, prepare and test output.
State files holding environment values are private. All participants run the
same fixed test entry and retain their existing node-specific environment.

Ray cannot safely control replacement of its own Python or CANN environment.
Image replay is rejected on Ray, as are mutations sharing Ray's Python prefix.
Use an independent Kubernetes exec channel for such transitions. Real cluster
switching still requires validation.

An interrupted test leaves `active-test.pending`. Confirm its services stopped
before clearing it. A disconnected transport does not prove remote work stopped.
Do not run concurrent bisectors against the same environment. The test command
must wait for distributed work and clean up its services before returning.

Existing shared-directory participants use `shared_directory.py` for barriers
and `../orchestration/shared_worker.py` for worker orchestration. These remain available for deployments
that already launch one bisect process per node.
