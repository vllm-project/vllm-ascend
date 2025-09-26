#! /bin/bash

# 定义要复制的文件或目录列表及其在目标主机上的对应路径
declare -A PATH_MAP=(
    ["/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.10/site-packages/vllm_npu/adaptor/envs.py"]="./examples/models/dsv3/vllm_npu_a3_0401/envs.py"
    ["/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.10/site-packages/vllm_npu/pta/model_executor/layers/fused_moe/layer.py"]="./examples/models/dsv3/vllm_npu_a3_0401/layer.py"
    ["/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.10/site-packages/vllm_npu/pta/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py"]="./examples/models/dsv3/vllm_npu_a3_0401/compressed_tensors_moe.py"
    ["/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.10/site-packages/vllm_npu/pta/model_executor/layers/fused_moe/fused_moe.py"]="./examples/models/dsv3/vllm_npu_a3_0401/fused_moe.py"
    ["/home/ma-user/anaconda3/envs/PyTorch-2.1.0/lib/python3.10/site-packages/vllm_npu/fx/worker/fx_worker.py"]="./examples/models/dsv3/vllm_npu_a3_0401/fx_worker.py"
)

# 循环遍历每个文件或目录，并进行复制
for item in "${!PATH_MAP[@]}"
do
    target_path="${PATH_MAP[$item]}"
    cp -r "$item" "$target_path"
    if [ $? -eq 0 ]; then
        echo "Successfully copied $item to $target_path."
    else
        echo "Failed to copy $item."
    fi
done
