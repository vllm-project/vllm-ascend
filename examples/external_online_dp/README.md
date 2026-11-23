Here is an example guiding how to use `launch_online_dp.py` to launch external dp vLLM servers. User can easily launch external dp servers following the steps below:

### Modify parameters in `run_dp_template.sh`

`run_dp_template.sh` is a template script used to launch each data parallel (DP) vLLM instance separately. `launch_online_dp.py` starts one shell process per local DP rank. Parameters you need to set manually include:

1. The IP and socket_ifname of your machine. If running on multi-nodes, please make sure the scripts on each node has been set with correct IP and socket_ifname of that node.
2. vLLM serving related parameters including model_path and other configurations. The launcher sets the API port, DP parameters, and TP/PCP/PP sizes. All other vLLM parameters in the template are examples; adapt them to your model and deployment.

The launcher always passes nine positional arguments to the template:

```text
$1 visible devices       $2 API port          $3 global DP size
$4 global DP rank        $5 DP address        $6 DP RPC port
$7 TP size               $8 PCP size          $9 PP size
```

Custom templates must read both `$8` and `$9` to use PCP and PP together. A template
that previously used `$8` for PP must read `$9` instead. The Python launcher
defaults omitted `--pcp-size` and `--pp-size` options to 1 and always passes both
values. The shell template requires all nine arguments; when calling it directly,
append `1 1` to an old seven-argument command to keep PCP and PP disabled.

### Run `launch_online_dp.py` with CL arguments

All the arguments that can be set by users are:

1. `--dp-size`: global data parallel size, must be set
2. `--tp-size`: tensor parallel size, default 1
3. `--dp-size-local`: local data parallel size, defaults to `dp_size`
4. `--dp-rank-start`: Starting rank for data parallel, default 0
5. `--dp-address`: IP address of data parallel master node
6. `--dp-rpc-port`: Port of data parallel master node, default 12345
7. `--vllm-start-port`: Starting port of vLLM serving instances, default 9000
8. `--pcp-size`: prefill context parallel size, default 1
9. `--pp-size`: pipeline parallel size, default 1
10. `--template` (alias `--script`): shell template path, default `./run_dp_template.sh`

Each local DP rank uses `TP × PCP × PP` consecutive devices. Device numbering starts
at 0 on each node, independent of `--dp-rank-start`. This launcher requires each
rank's TP/PCP/PP workers to fit on one node; multi-node DP uses the rank offsets
shown below. DCP reuses existing workers and is configured in the template rather
than multiplied into the device count.

Template paths and paths used inside the template are relative to the current
working directory. Templates should run vLLM in the foreground. The launcher
checks every local rank's exit status every 0.1 seconds. A nonzero exit from any
rank triggers cleanup of the recorded rank process groups and a nonzero launcher
exit; a running rank cannot hide another rank's failure. The launcher succeeds
only after all ranks exit successfully. Bash `pipefail` also preserves failures
when a template pipes vLLM output through `tee`.

On Linux, each rank runs in its own process group. If starting or monitoring
ranks raises an exception, including a detected rank failure or
`KeyboardInterrupt` from `Ctrl+C`, the launcher sends `SIGTERM` to every recorded
group, waits for its shell, then
re-raises the exception. Templates must keep their workers in the same process
group; do not daemonize them or use `setsid` inside the template.

Receiving `SIGTERM` raises `SystemExit` and enters the same cleanup path. After
cleanup completes, the launcher exits with status 0 for this requested stop.

This is best-effort cleanup: it has no timeout or forced termination.
Waiting covers the shells; descendants
must respond to `SIGTERM` themselves. It does not guarantee cleanup if a stop
signal arrives before `Popen` returns or if cleanup is interrupted again.

An example of running external DP in one single node:

```bash
cd examples/external_online_dp
# running DP4 TP4 in a node with 16 NPUs
python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address x.x.x.x --dp-rpc-port 12342
```

An example of running external DP in two nodes:

```bash
cd examples/external_online_dp
# running DP4 TP4 in two nodes with 8 NPUs each

# On node 0:
python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 2 --dp-rank-start 0 --dp-address x.x.x.x --dp-rpc-port 12342

# On node 1:
python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 2 --dp-rank-start 2 --dp-address x.x.x.x --dp-rpc-port 12342
```

An example combining PCP and PP on a node with 8 NPUs:

```bash
cd examples/external_online_dp
export VLLM_USE_V2_MODEL_RUNNER=1
python launch_online_dp.py \
    --dp-size 2 --tp-size 1 --pcp-size 2 --pp-size 2 \
    --dp-address x.x.x.x --vllm-start-port 9000
```

This assigns devices `0,1,2,3` to DP rank 0 and `4,5,6,7` to DP rank 1,
with API ports 9000 and 9001. External load balancers use these API endpoints, not the
DP RPC port. PCP requires Model Runner V2 on Ascend; the model and other enabled
features must support the selected combination. See the
[context parallel guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/context_parallel.html)
and [pipeline parallel guide](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/pipeline_parallel.html).
