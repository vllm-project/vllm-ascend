# Elastic Milestone 1 Example

## Deepseek-V2-Lite manual scaling and inference
**Set model path: in all bash scripts `export MODEL_PATH=/path/to/your/model`**

### In 4 terminals
Terminal 1.
- Start Ray `bash vllm_ascend/elastic_scaling/_start_ray.sh`
- Deploy WORM HMM server `bash vllm_ascend/elastic_scaling/_ds_v2_worm_deploy_server.sh`

Terminal 2. 
- Start IMM inference instance #1 (DP1TP4EP4) `bash vllm_ascend/elastic_scaling/_ds_v2_elastic_d1t4e4.sh`

Terminal 3. 
- Start IMM inference instance #2 (DP2TP4EP8) `bash vllm_ascend/elastic_scaling/_ds_v2_elastic_d2t4e8.sh`

Terminal 4. 
- Zero-copy and prepare instance #1 for serving `python vllm_ascend/elastic_scaling/_scaling_req.py --num_scale_units 0 --tp 4 --inference_port 7101`
- Send request `python vllm_ascend/elastic_scaling/_inference_req.py --port 7101`
- Zero-copy and prepare instance #2 for serving `python vllm_ascend/elastic_scaling/_scaling_req.py --num_scale_units 1 --tp 4 --inference_port 7102`
- Send request `python vllm_ascend/elastic_scaling/_inference_req.py --port 7102`


## Qwen3-30B-A3B manual scaling and inference
**Set model path: in all bash scripts `export MODEL_PATH=/path/to/your/model`**

### In 4 terminals
Terminal 1.
- Start Ray `bash vllm_ascend/elastic_scaling/_start_ray.sh`
- Deploy WORM HMM server `bash vllm_ascend/elastic_scaling/_qwen_worm_deploy_server.sh`

Terminal 2. 
- Start IMM inference instance #1 (DP1TP4EP4) `bash vllm_ascend/elastic_scaling/_qwen_elastic_d1t4e4.sh`

Terminal 3. 
- Start IMM inference instance #2 (DP2TP4EP8) `bash vllm_ascend/elastic_scaling/_qwen_elastic_d2t4e8.sh`

Terminal 4. 
- Zero-copy and prepare instance #1 for serving `python vllm_ascend/elastic_scaling/_scaling_req.py --num_scale_units 0 --tp 4 --inference_port 7101`
- Send request `python vllm_ascend/elastic_scaling/_inference_req.py --port 7101`
- Zero-copy and prepare instance #2 for serving `python vllm_ascend/elastic_scaling/_scaling_req.py --num_scale_units 1 --tp 4 --inference_port 7102`
- Send request `python vllm_ascend/elastic_scaling/_inference_req.py --port 7102`
