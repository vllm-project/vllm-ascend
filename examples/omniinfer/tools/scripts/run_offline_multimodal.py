import os
import time
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import numpy as np
import cv2

os.environ.update({
    "GLOO_SOCKET_IFNAME": "enp23s0f3",
    "TP_SOCKET_IFNAME": "enp23s0f3",
    "VLLM_USE_V1": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "fork",
    "ASCEND_LAUNCH_BLOCKING": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_INTRA_ROCE_ENABLE": "1",
    "HCCL_INTRA_PCIE_ENABLE": "0",
})

# User Configuration
MODEL_TYPE = "internvl" # Model type: "internvl" or "qwen2_vl"
MODEL_PATH = "/data/models/InternVL2_5-8B"

# Test requests list - each request contains (filepath, question, type)
TEST_REQUESTS = [
    ("/your_path/image1.jpg", "请描述一下图片的内容", "image"),
    ("/your_path/image2.jpg", "图像中有什么东西", "image"),
    ("/your_path/video1.mp4", "请描述这个视频中发生了什么", "video"),
    ("", "请介绍一下人工智能的发展历程", "text")

]

def create_requests():
    tokens = {
        "internvl": {"image": "<image>", "video": "<video>"},
        "qwen2_vl": {"image": "<|image_pad|>", "video": "<|video_pad|>"}
    }
    requests = []
    for file_path, prompt, req_type in TEST_REQUESTS:
        # Add corresponding tokens for multimodal requests
        if req_type in tokens[MODEL_TYPE]:
            prompt = tokens[MODEL_TYPE][req_type] + prompt
        requests.append((file_path, prompt, req_type))
    
    return requests

requests = create_requests()

print("Initializing model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.7,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 1, "video": 1},
)
sampling_params = SamplingParams(max_tokens=200, temperature=0.0)

def load_video(video_path, num_frames=8):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)

    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    cap.release()
    return np.array(frames) if frames else None

def create_input(file_path, prompt, media_type):
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    if media_type == "text":
        return {"prompt": formatted_prompt}
    elif media_type == "image":
        return {"prompt": formatted_prompt, "multi_modal_data": {"image": Image.open(file_path)}}
    elif media_type == "video":
        return {"prompt": formatted_prompt, "multi_modal_data": {"video": load_video(file_path)}}


print("=== Multimodal Test ===")
inputs = []

for file_path, prompt, media_type in requests:
    try:
        if media_type != "text" and not os.path.exists(file_path):
            print(f"Skip: File does not exist {file_path}")
            continue
        
        input_data = create_input(file_path, prompt, media_type)
        if input_data:
            inputs.append(input_data)
            print(f"Loaded {media_type} request")
    except Exception as e:
        print(f"Loading failed: {e}")

if inputs:
    print(f"\nStarting to process {len(inputs)} requests...")
    start_time = time.time()

    outputs = llm.generate(inputs, sampling_params=sampling_params)

    print(f"Completed, time taken: {time.time() - start_time:.2f} seconds\n")

    # Display results
    for i, (output, (_, prompt, media_type)) in enumerate(zip(outputs, requests), 1):
        print(f"=== Request {i} ({media_type}) ===")
        print(f"Question: {prompt}")
        print(f"Answer: {output.outputs[0].text}\n")
else:
    print("No valid requests")
