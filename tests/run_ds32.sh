export HCCL_OP_EXPANSION_MODE="AIV"
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_MLAPO="1"
export ASCEND_RT_VISIBLE_DEVICES="0,1,2,3"

python3 -c "
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model='/home/weights/DeepSeek-V3.2-W8A8-Pruning',
    tensor_parallel_size=4,
    data_parallel_size=1,
    max_model_len=65536,
    max_num_batched_tokens=4096,
    trust_remote_code=True,
    quantization='ascend',
    gpu_memory_utilization=0.9,
    enable_expert_parallel=True,
    enforce_eager=True
)

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=2048
)

# 输入提示词
prompts = ['请你介绍一下李白']

# 执行推理
outputs = llm.generate(prompts, sampling_params)

# 输出结果
for output in outputs:
    print(f'提示词: {output.prompt}')
    print(f'生成结果: {output.outputs[0].text}')
    print('-' * 80)
"