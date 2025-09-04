Here is an example guiding how to use `launch_online_dp.py` to launch external dp server in vllm. User can easily launch external dp server following the steps below:

### Modify parameters in `run_dp_template.sh`
`run_dp_template.sh` is an template script used to launch each dp vllm instance separately. It will be called by `launch_online_dp.py` in multi threads and most of its configurations are set by `launch_online_dp.py`. Parameters you need to set manually include:

1. The IP and socket_ifname of your machine. If running on multi-nodes, please make sure the scripts on each node has been set with correct IP and socket_ifname of that node.
2. vLLM serving related parameters including model_path and other configurations. Note that port, dp-related parammeters and tp_size is set by `launch_online_dp.py`, all the other vLLM parameters in this file only serve as an example and you are free to modify them according to your purpose.

### Run `launch_online_dp.py` with CL arguments
All the arguments that can be set by users are:

1. `--dp-size`: global data parallel size, must be set
2. `--tp-size`: tensor parallel size, default 1
3. `--dp-size-local`: local data parallel size, defaultly set to `dp_size`
4. `--dp-rank-start`: Starting rank for data parallel, default 0
5. `--dp-address`: IP address of data parallel master node
6. `--dp-rpc-port`: Port of data parallel master node, default 12345
7. `--vllm-start-port`: Starting port of vLLM serving instances, default 9000

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

