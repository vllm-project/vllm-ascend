Here is an example guiding how to use `launch_online_dp.py` to launch external dp vLLM servers. User can easily launch external dp servers following the steps below:

### Modify parameters in `run_dp_template.sh`

`run_dp_template.sh` is a template script used to launch each data parallel (dp) vLLM instance separately. It will be called by `launch_online_dp.py` in multiple threads and most of its configurations are set by `launch_online_dp.py`. Parameters you need to set manually include:

1. The IP and socket_ifname of your machine. If running on multi-nodes, please make sure the scripts on each node has been set with correct IP and socket_ifname of that node.
2. vLLM serving related parameters including model_path and other configurations. Note that port, dp-related parameters, tp_size and pcp_size are set by `launch_online_dp.py`, all the other vLLM parameters in this file only serve as an example and you are free to modify them according to your purpose.

`launch_online_dp.py` passes the following positional arguments to `run_dp_template.sh`:

| Position | Value | Used by the template |
|----------|-------|----------------------|
| `$1` | visible devices | `ASCEND_RT_VISIBLE_DEVICES` |
| `$2` | API port of this rank | `--port` |
| `$3` | global data parallel size | `--data-parallel-size` |
| `$4` | data parallel rank | `--data-parallel-rank` |
| `$5` | DP master node address | `--data-parallel-address` |
| `$6` | DP master node RPC port | `--data-parallel-rpc-port` |
| `$7` | tensor parallel size | `--tensor-parallel-size` |
| `$8` | prefill context parallel size | `--prefill-context-parallel-size` |

### Run `launch_online_dp.py` with CL arguments

All the arguments that can be set by users are:

1. `--dp-size`: global data parallel size, must be set
2. `--tp-size`: tensor parallel size, default 1
3. `--pcp-size`: prefill context parallel size, default 1. Each data parallel rank consumes `tp-size * pcp-size` logical devices.
4. `--dp-size-local`: local data parallel size, defaultly set to `dp_size`
5. `--dp-rank-start`: Starting rank for data parallel, default 0
6. `--dp-address`: IP address of data parallel master node
7. `--dp-rpc-port`: Port of data parallel master node, default 12345
8. `--vllm-start-port`: Starting port of vLLM serving instances, default 9000

An example of running external DP in one single node:

```(python)
cd examples/external_online_dp
# running DP4 TP4 in a node with 16 NPUs
python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 4 --dp-rank-start 0 --dp-address x.x.x.x --dp-rpc-port 12342
```

An example of running external DP in two nodes:

```(python)
cd examples/external_online_dp
# running DP4 TP4 in two nodes with 8 NPUs each

# On node 0:
python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 2 --dp-rank-start 0 --dp-address x.x.x.x --dp-rpc-port 12342

# On node 1:
python launch_online_dp.py --dp-size 4 --tp-size 4 --dp-size-local 2 --dp-rank-start 2 --dp-address x.x.x.x --dp-rpc-port 12342
```

An example of running external DP with prefill context parallelism:

```(python)
cd examples/external_online_dp
# running DP2 PCP4 in a node with 8 NPUs; each DP rank uses tp-size * pcp-size = 4 devices
python launch_online_dp.py --dp-size 2 --tp-size 1 --pcp-size 4 --dp-size-local 2 --dp-rank-start 0 --dp-address x.x.x.x --dp-rpc-port 12342
```
