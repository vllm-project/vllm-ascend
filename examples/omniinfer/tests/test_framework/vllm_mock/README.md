## vLLM Mock Model 
Mock model for vllm testing without running the actual model, to verify (v) the pre- and postprocessing:

![image](./docs/mock_model.png 'mock_model.png')

The replay of captured outputs can be performed and is faster than running the actual model. Moreover, for replay it may suffice to use less NPUs than needed for the actual model.


## Automated Install
No need to patch anything anymore. 

You can run any scripts in the ./scripts folder, see also further below for more instructions

## Configs
The mock model has a random output mode, capture mode and replay mode, as well as PD separation (KV_CACHE_MODE) and no-NPU support, with the configs set by environment variables (there are some more variables, check in mock.py for your needs):

```python
# COMMENT OUT the ones not in use. IMPORTANT: Make sure you use temperature=0!
# os.environ["KV_CACHE_MODE"] = "1"  # for PD separation
# os.environ["CAPTURE_MODE"] = "1"  # capture inputs and outputs to cache
# os.environ["REPLAY_MODE"] = "1"  # replay inputs and outputs from cache
os.environ["RANDOM_MODE"] = "1"
# os.environ["FORWARD_TIME"] = "25"  # 25 milliseconds
# os.environ["SIMULATE_ELAPSED_TIME"] = "1"  # replay model output after waiting for approx. captured time.
# os.environ["MOCK_CAPTURE_DIR"] = "/home/kc/capture/"  # saving folder for logs of inputs and outputs
# os.environ["MOCK_CAPTURE_FILE"] = ".mock"
# os.environ["MOCK_CAPTURE_FILE_LOCK"] = ".lock"
```

- When CAPTURE_MODE (set), the model outputs for each prompt (identified by their prompt token ids) are captured to MOCK_CAPTURE_FILE in MOCK_CAPTURE_DIR.
- When instead REPLAY_MODE (set), the MOCK_CAPTURE_FILE in MOCK_CAPTURE_DIR is read into a cache, and whenever a request with matching prompt token ids comes in, the corresponding position outputs are replayed.
- KV_CACHE_MODE needed for PD separation, captures or replays the KV cache on the prefill nodes.
- RANDOM_MODE is very fast and outputs random outputs regardless of input, with FORWARD_TIME simulated time for computing output.
- SIMULATE_ELAPSED_TIME on REPLAY_MODE will simulate time for computing output as it took on the NPUs.
- PREFILL_PROCESS must be set for the P node of PD separation
- MOCK_COMPUTE_LOGITS allows also mocking logits if needed (may be slow because logits are high-dimensional) 
- TORCH_COMPILE_MODE_MOCK can be set for using torch graph compile mode, in case automatic detection fails. Does not support sleeping for FORWARD_TIME.

## Run (Offline Mode)
Set capture_mode and not replay_mode, register the mock model instead of the original model in vllm_ascend/models/init.py and then run (with temperature 0 and use tp=2, otherwise accuracy drops due to an unrelated issue) to capture / replay the outputs or produce random ones. See the following scripts:
```bash
python ./scripts/random_mock_model_tp2.py
python ./scripts/capture_mock_model_tp2.py
python ./scripts/replay_mock_model_tp2.py
```

## Run (Single-Node Online API-Server)
To run in serving mode on a single node, register the mock model and then serve the mock model in either capture mode or replay mode, similar to offline mode. See:
```bash
python ./scripts/vllm_serve.py
```
and send some prompts to the server
```bash
bash ./scripts/prompt.sh
```

## Run (Multi-Node Online API-Server)
To run multi-node, simultaneously run ./scripts/vllm_serve_multinode_master.sh and ./scripts/vllm_serve_multinode_slave.sh on nodes 0 and 1 respectively. Make sure to change the IPs in the script to the master node IP, and have access to the same files / copies of the same files with the same path. The server will start up on the master node. 

Note: Make sure to change configs on both machines / in both master & slave scripts. Make sure also to change all of the desired environment variables before calling ray start in the script!

## Run (API-Servers  PD Separation)
To run PD separation on one node (1P1D), set the PREFILL_PROCESS=1 environment variable for the P node, and KV_CACHE_MODE=1 for both P/D nodes. 

Apart from that, it suffices to set the environment variables as usual, see the script for reference. 

Note: For PD separation, capture / replay on decode node works, but will not verify the KV cache input on the decode node due to numerical inaccuracy issues from layer normalization, since the KV cache will look different on every run.

Please see the following config as an example:
```python
os.environ["PREFILL_PROCESS"] = "1"  # ONLY ON P NODE!
os.environ["RANDOM_MODE"] = "1"  # replay inputs and outputs from the cache
# os.environ["SIMULATE_ELAPSED_TIME"] = "1"  # replay model output after waiting for approx. captured time.
os.environ["KV_CACHE_MODE"] = "1"  # capture inputs and outputs, for use with the replaying mode.
# os.environ["CAPTURE_MODE"] = "1"  # capture inputs and outputs, for use with the replaying mode.
# os.environ["REPLAY_MODE"] = "1"  # capture inputs and outputs, for use with the replaying mode.
# os.environ["MOCK_CAPTURE_DIR"] = "/home/kc/capture/"  # saving folder for logs of inputs and outputs, ensure this exists
# os.environ["MOCK_CAPTURE_FILE"] = ".mock_cache_pd"
# os.environ["MOCK_CAPTURE_FILE_LOCK"] = ".lock"
```

## Run (Single-Node Online API-Server without NPU)
To run without NPUs, installation from zero as a script can be found in ./bash_setup_full_mock_no_npus.sh.

Then start the API server (set ASCEND_RT_VISIBLE_DEVICES as if you had enough NPUs, and also set env variable NO_NPU_MOCK=1):
```bash
python ./scripts/vllm_serve_no_npu.py
```


## Limitations
- Sometimes the accuracy of model seems to drop, in particular for TP=1. It appears an unrelated issue, you can easily test by removing the mock model in init.py.
- For PD separation, capture / replay on decode node works, but will not verify the KV cache input on the decode node due to numerical inaccuracy issues from layer normalization, since the KV cache will look different on every run. Also, without NPU the PD separation won't work properly, since the llm_datadist package directly works with CANN / Ascend stack