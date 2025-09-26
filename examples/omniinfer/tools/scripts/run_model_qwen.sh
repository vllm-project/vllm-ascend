# Runs Qwen model in default combined deployment mode 
# Uses standard execution (no graph optimization) with full NPU utilization
python run_model_qwen.py \
    --model-path /data/models/QwQ-32B \
    --deploy-mode default \
    --graph-true 'false' \
    --model-name qwen \
    --server-list 0,1,2,3,4,5,6,7 \
    --network-interface enp23s0f3 \
    --host-ip 7.150.12.75 \
    --https-port 8001

# Runs Qwen model in default deployment with graph optimization
# Maximizes throughput using all NPUs (0-7) with accelerated execution
# python run_model_qwen.py \
#     --model-path /data/models/QwQ-32B \
#     --deploy-mode default \
#     --graph-true 'true' \
#     --model-name qwen \
#     --server-list 0,1,2,3,4,5,6,7 \
#     --network-interface enp23s0f3 \
#     --host-ip 7.150.12.75 \
#     --https-port 8001

# Runs Qwen in split deployment mode (Prefill and Decoder on separate devices)
# Uses standard execution and custom service port 
# python run_model_qwen.py \
#     --model-path /data/models/QwQ-32B \
#     --deploy-mode pd_separate \
#     --graph-true 'false' \
#     --model-name qwen \
#     --network-interface enp23s0f3 \
#     --prefill-server-list 0,1,2,3,4,5,6,7 \
#     --decode-server-list 8,9,10,11,12,13,14,15 \
#     --host-ip 7.150.12.75 \
#     --https-port 8001 \
#     --service-port 6660


# Runs Qwen in split deployment mode (Prefill and Decoder on separate devices)
# Uses standard execution and custom service port 
# python run_model_qwen.py \
#     --model-path /data/models/QwQ-32B \
#     --deploy-mode pd_separate \
#     --graph-true 'true' \
#     --model-name qwen \
#     --network-interface enp23s0f3 \
#     --prefill-server-list 0,1,2,3,4,5,6,7 \
#     --decode-server-list 8,9,10,11,12,13,14,15 \
#     --host-ip 7.150.12.75 \
#     --https-port 8001 \
#     --service-port 6660

