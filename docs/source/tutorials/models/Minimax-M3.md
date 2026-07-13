# Minimax M3

## Environment Preparation

### Model Weight

`Minimax-m3`(BF16 version): requires 1 Ascend 910C (with 16 x 64G NPUs). [Download model weight](https://www.modelscope.cn/collections/MiniMax/MiniMax-M3)

It is recommended to place the model weight in a shared cache directory.

### Installation

- Step 1： Download v0.21.0rc1 Docker image
  ```
  docker pull quay.io/ascend/vllm-ascend:v0.21.0rc1-a3
  ```

- Step 2: Start Docker container
  ```
  # 更新 vllm-ascend 镜像，并配置对应的Image名
  export IMAGE=quay.io/ascend/vllm-ascend:v0.21.0rc1-a3
  export NAME=minimax-m3-dev

  # 使用定义的变量运行容器
  # 根据您的设备更新 --device（Atlas A3：/dev/davinci[0-15] Atlas A2：/dev/davinci[0-7]）。
  # 注意：若使用 Docker 桥接网络，请提前开放可供多节点通信的端口
  docker run --rm \
  --name $NAME \
  --net=host \
  --shm-size=100g \
  --device /dev/davinci0 \
  --device /dev/davinci1 \
  --device /dev/davinci2 \
  --device /dev/davinci3 \
  --device /dev/davinci4 \
  --device /dev/davinci5 \
  --device /dev/davinci6 \
  --device /dev/davinci7 \
  --device /dev/davinci8 \
  --device /dev/davinci9 \
  --device /dev/davinci10 \
  --device /dev/davinci11 \
  --device /dev/davinci12 \
  --device /dev/davinci13 \
  --device /dev/davinci14 \
  --device /dev/davinci15 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v /root/.cache:/root/.cache \
  -it $IMAGE bash
  ```

- Step 3: Update vLLM
  ```
  cd /vllm-workspace/vllm
  git checkout v0.24.0

  # Install _rust_tool_parser for the Rust frontend.
  pip install setuptools-rust
  ./build_rust.sh
  ```

- Step 4: Update vLLM Ascend
  ```
  cd /vllm-workspace/vllm-ascend
  git fetch origin pull/10682/merge:pr-10682
  git merge pr-10682
  ```

## Deployment

Start the online serving service with the following command:

- For BF16 version

  ```
  export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
  export HCCL_OP_EXPANSION_MODE="AIV"
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

  vllm serve ${WEIGHT_PATH} \
    --served-model-name minimax-m3 \
    --trust-remote-code \
    --max-model-len 43008 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --max-num-seqs 16 \
    --distributed_executor_backend "mp" \
    --gpu-memory-utilization 0.92 \
    --reasoning-parser minimax_m3 \
    --limit-mm-per-prompt '{"image":1}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"enable_static_kernel": true, "fuse_norm_quant":false}, "multistream_overlap_shared_expert": true, "weight_nz_mode": 2}' \
    --port 11223 > ${LOG_PATH} 2>&1 &
  ```

- For W8A8 version
  ```
  export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
  export HCCL_OP_EXPANSION_MODE="AIV"
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

  vllm serve ${WEIGHT_PATH} \
  --served-model-name minimax-m3 \
  --trust-remote-code \
  --max-model-len 131072 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --max-num-seqs 16 \
  --distributed_executor_backend "mp" \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser minimax_m3 \
  --limit-mm-per-prompt '{"image":1}' \
  --speculative-config '{"model":"${EAGLE3_WEIGHT_PATH}", "method":"eagle3", "num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"enable_static_kernel": true, "fuse_norm_quant": true}, "multistream_overlap_shared_expert": true, "weight_nz_mode": 2}' \
  --port 11223 > ${LOG_PATH} 2>&1 &
  ```

**Note**: In the script above, `max-num-seqs` is set to 16, which represents the maximum number of sequences the scheduler can process in a single iteration. Adjust the `max-num-seqs` parameter dynamically based on actual business.

For text-only deployment, `--limit-mm-per-prompt` can be omitted. For multimodal deployment, configure this parameter according to the actual request shape. For example, use `--limit-mm-per-prompt '{"image":2}'` for two-image requests, and use `--limit-mm-per-prompt '{"video":1}'` for one-video requests.

### A2 Deployment Examples

The examples below use Ascend A2 servers. Update `WEIGHT_PATH`, `EAGLE3_WEIGHT_PATH`, `LOG_PATH`, `local_ip`, `node0_ip`, and `IFNAME` based on the actual environment.

- For BF16 version on 2 A2 nodes

  Run the following command on node 0:

  ```
  local_ip="${NODE0_IP}"
  node0_ip="${NODE0_IP}"

  export HCCL_IF_IP=$local_ip
  export IFNAME="${NETWORK_INTERFACE}"
  export GLOO_SOCKET_IFNAME="$IFNAME"
  export TP_SOCKET_IFNAME="$IFNAME"
  export HCCL_SOCKET_IFNAME="$IFNAME"
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export VLLM_ENGINE_READY_TIMEOUT_S=3600
  export HCCL_CONNECT_TIMEOUT=7200
  export ASCEND_CONNECT_TIMEOUT=10000
  export ASCEND_TRANSFER_TIMEOUT=10000
  export VLLM_RPC_TIMEOUT=1800000
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export HCCL_OP_EXPANSION_MODE="AIV"
  export TASK_QUEUE_ENABLE=1
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

  vllm serve ${WEIGHT_PATH} \
    --host 0.0.0.0 \
    --served-model-name minimax-m3 \
    --trust-remote-code \
    --max-model-len 40960 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address $node0_ip \
    --distributed_executor_backend "mp" \
    --gpu-memory-utilization 0.94 \
    --reasoning-parser minimax_m3 \
    --limit-mm-per-prompt '{"image":1}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"enable_static_kernel": true, "fuse_norm_quant":false}, "multistream_overlap_shared_expert": true, "weight_nz_mode": 2}' \
    --port 11223 > ${LOG_PATH} 2>&1 &
  ```

  Run the following command on node 1:

  ```
  local_ip="${NODE1_IP}"
  node0_ip="${NODE0_IP}"

  export HCCL_IF_IP=$local_ip
  export IFNAME="${NETWORK_INTERFACE}"
  export GLOO_SOCKET_IFNAME="$IFNAME"
  export TP_SOCKET_IFNAME="$IFNAME"
  export HCCL_SOCKET_IFNAME="$IFNAME"
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export VLLM_ENGINE_READY_TIMEOUT_S=3600
  export HCCL_CONNECT_TIMEOUT=7200
  export ASCEND_CONNECT_TIMEOUT=10000
  export ASCEND_TRANSFER_TIMEOUT=10000
  export VLLM_RPC_TIMEOUT=1800000
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export HCCL_OP_EXPANSION_MODE="AIV"
  export TASK_QUEUE_ENABLE=1
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

  vllm serve ${WEIGHT_PATH} \
    --host 0.0.0.0 \
    --served-model-name minimax-m3 \
    --trust-remote-code \
    --headless \
    --max-model-len 40960 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address $node0_ip \
    --distributed_executor_backend "mp" \
    --gpu-memory-utilization 0.94 \
    --reasoning-parser minimax_m3 \
    --limit-mm-per-prompt '{"image":1}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"enable_static_kernel": true, "fuse_norm_quant":false}, "multistream_overlap_shared_expert": true, "weight_nz_mode": 2}' \
    --port 11223 > ${LOG_PATH} 2>&1 &
  ```

- For W8A8 version on 2 A2 nodes

  Run the following command on node 0:

  ```
  local_ip="${NODE0_IP}"
  node0_ip="${NODE0_IP}"

  export HCCL_IF_IP=$local_ip
  export IFNAME="${NETWORK_INTERFACE}"
  export GLOO_SOCKET_IFNAME="$IFNAME"
  export TP_SOCKET_IFNAME="$IFNAME"
  export HCCL_SOCKET_IFNAME="$IFNAME"
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export VLLM_ENGINE_READY_TIMEOUT_S=3600
  export HCCL_CONNECT_TIMEOUT=7200
  export ASCEND_CONNECT_TIMEOUT=10000
  export ASCEND_TRANSFER_TIMEOUT=10000
  export VLLM_RPC_TIMEOUT=1800000
  export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
  export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
  export HCCL_OP_EXPANSION_MODE="AIV"
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

  vllm serve ${WEIGHT_PATH} \
    --host 0.0.0.0 \
    --served-model-name minimax-m3 \
    --trust-remote-code \
    --max-model-len 131072 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 0 \
    --data-parallel-address $node0_ip \
    --distributed_executor_backend "mp" \
    --gpu-memory-utilization 0.92 \
    --reasoning-parser minimax_m3 \
    --limit-mm-per-prompt '{"image":1}' \
    --speculative-config '{"model":"${EAGLE3_WEIGHT_PATH}", "method":"eagle3", "num_speculative_tokens":3}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"enable_static_kernel": true, "fuse_norm_quant":false}, "multistream_overlap_shared_expert": false, "weight_nz_mode": 2}' \
    --port 11223 > ${LOG_PATH} 2>&1 &
  ```

  Run the following command on node 1:

  ```
  local_ip="${NODE1_IP}"
  node0_ip="${NODE0_IP}"

  export HCCL_IF_IP=$local_ip
  export IFNAME="${NETWORK_INTERFACE}"
  export GLOO_SOCKET_IFNAME="$IFNAME"
  export TP_SOCKET_IFNAME="$IFNAME"
  export HCCL_SOCKET_IFNAME="$IFNAME"
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export VLLM_ENGINE_READY_TIMEOUT_S=3600
  export HCCL_CONNECT_TIMEOUT=7200
  export ASCEND_CONNECT_TIMEOUT=10000
  export ASCEND_TRANSFER_TIMEOUT=10000
  export VLLM_RPC_TIMEOUT=1800000
  export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000
  export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
  export HCCL_OP_EXPANSION_MODE="AIV"
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:$LD_PRELOAD

  vllm serve ${WEIGHT_PATH} \
    --host 0.0.0.0 \
    --served-model-name minimax-m3 \
    --trust-remote-code \
    --headless \
    --max-model-len 131072 \
    --tensor-parallel-size 8 \
    --enable-expert-parallel \
    --max-num-seqs 8 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-start-rank 1 \
    --data-parallel-address $node0_ip \
    --distributed_executor_backend "mp" \
    --gpu-memory-utilization 0.92 \
    --reasoning-parser minimax_m3 \
    --limit-mm-per-prompt '{"image":1}' \
    --speculative-config '{"model":"${EAGLE3_WEIGHT_PATH}", "method":"eagle3", "num_speculative_tokens":3}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --additional-config '{"enable_cpu_binding":true, "ascend_compilation_config":{"enable_static_kernel": true, "fuse_norm_quant":false}, "multistream_overlap_shared_expert": false, "weight_nz_mode": 2}' \
    --port 11223 > ${LOG_PATH} 2>&1 &
  ```

## Thinking Mode

MiniMax-M3 supports three thinking modes, controlled via `thinking_mode` in `chat_template_kwargs`:

| Mode | Behavior | Use Case |
|------|----------|----------|
| `enabled` | The model thinks before every response, including after tool results | Complex reasoning, agents |
| `disabled` | No thinking; the model answers directly | Latency-sensitive turns |
| `adaptive` | The model decides whether to think based on the task (default when unset) | General use |

### Request Examples

**With thinking disabled (curl):**

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m3",
    "messages": [{"role": "user", "content": "who are you?"}],
    "max_tokens": 100,
    "stream": false,
    "top_p": 0.95,
    "top_k": 40,
    "temperature": 1.0,
    "chat_template_kwargs": {"thinking_mode": "disabled"}
  }'
```

Change `"thinking_mode"` to `"enabled"` or `"adaptive"` as needed. The deprecated `enable_thinking` parameter (equivalent to `thinking_mode: "enabled"`) is also supported.

**With thinking enabled (Python SDK):**

```python
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="minimax-m3",
    messages=[{"role": "user", "content": "Prove there are infinitely many primes."}],
    extra_body={"chat_template_kwargs": {"thinking_mode": "enabled"}},
)
msg = response.choices[0].message
print(getattr(msg, "reasoning", None))  # the <mm:think> block
print(msg.content)                       # the final answer
```


## Reasoning Parser

The MiniMax-M3 reasoning parser (`--reasoning-parser minimax_m3`) extracts the thinking block `<mm:think>...</mm:think>` from model output and exposes it as the `reasoning` field. The remaining text is returned as `content`.

### Server Configuration

The `--reasoning-parser minimax_m3` flag enables the MiniMax-M3 reasoning parser, which splits model output into reasoning and content using `<mm:think>...</mm:think>` delimiters:

```bash
vllm serve ${WEIGHT_PATH} \
  --reasoning-parser minimax_m3 \
  ...
```

### Output Format

MiniMax-M3 uses explicit thinking delimiters:

```
<mm:think>reasoning process...</mm:think>final answer
```

### Parser Behavior

- **`thinking_mode="enabled"`**: The chat template pre-fills `<mm:think>` in the prompt. Generated text starts inside the reasoning block and transitions to content after `</mm:think>`.
- **`thinking_mode="disabled"` or default**: Model output is treated as plain content. If `<mm:think>` appears, the parser splits on the delimiters.
- **Streaming**: Reasoning and content are streamed incrementally via `DeltaMessage.reasoning` and `DeltaMessage.content` token-by-token.
- **Token counting**: Reasoning tokens inside `<mm:think>` blocks are correctly counted.

## Tool Call Parser

MiniMax-M3 uses a namespace-delimited XML format for tool calls. Enable it with `--tool-parser minimax_m3`.

### Server Configuration

When both `--reasoning-parser minimax_m3` and `--tool-call-parser minimax_m3` are specified, the parsers work together automatically to handle responses that contain both reasoning blocks and tool calls:

```bash
vllm serve ${WEIGHT_PATH} \
  --reasoning-parser minimax_m3 \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m3 \
  ...
```

### Tool Call Format

Each structural tag is preceded by the `]<]minimax[>[` namespace marker:

```xml
]<]minimax[>[<tool_call>
]<]minimax[>[<invoke name="create_order">
]<]minimax[>[<user_id>42]<]minimax[>[</user_id>
]<]minimax[>[<shipping>
]<]minimax[>[<city>Singapore]<]minimax[>[</city>
]<]minimax[>[<zip>018956]<]minimax[>[</zip>
]<]minimax[>[</shipping>
]<]minimax[>[</invoke>
]<]minimax[>[</tool_call>
```

### Key Features

- **Recursive parameter parsing**: Supports nested objects and arrays (e.g., `shipping` containing `city`/`zip`).
- **Schema-aware type coercion**: String parameter values are automatically converted to the correct types (integer, boolean, object, array) based on the function's JSON Schema definition.
- **Multiple invocations**: A single `<tool_call>` block can contain multiple `<invoke>` blocks.
- **Streaming**: Tool name and argument fragments are streamed incrementally as the `<invoke>` block is received.

### Request Example (curl)

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimax-m3",
    "messages": [{"role": "user", "content": "What's the weather like in Shanghai?"}],
    "max_tokens": 300,
    "stream": false,
    "tool_choice": "auto",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City or country name"
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": false
                }
            }
        }
    ],
    "chat_template_kwargs": {"thinking_mode": "disabled"}
  }'
```

## Functional Verification

- Text

  ```
  #!/bin/bash
  # Verify MiniMax M3 vLLM service via OpenAI-compatible chat completions API.
  # Usage: bash script/verify_curl.sh
  
  set -euo pipefail
  
  BASE_URL="${BASE_URL:-http://127.0.0.1:11223}"
  MODEL="${MODEL:-minimax-m3}"
  PROMPT="Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nA student regrets that he fell asleep during a lecture in electrochemistry, facing the following incomplete statement in a test:\nThermodynamically, oxygen is a …… oxidant in basic solutions. Kinetically, oxygen reacts …… in acidic solutions.\nWhich combination of weaker/stronger and faster/slower is correct?\n\nA) weaker – faster\nB) stronger – faster\nC) weaker - slower\nD) stronger – slower"
  # PROMPT='On $\\triangle ABC$ points $A,D,E$, and $B$ lie that order on side $\\overline{AB}$ with $AD=4, DE=16$, and $EB=8$. Points $A,F,G$, and $C$ lie in that order on side $\\overline{AC}$ with $AF=13, FG=52$, and $GC=26$. Let $M$ be the reflection of $D$ through $F$, and let $N$ be the reflection of $G$ through $E$. Quadrilateral $DEGF$ has area 288. Find the area of heptagon $AFNBCEM$.\nRemember to put your final answer within \\boxed{}'
  MAX_TOKENS="${MAX_TOKENS:-8888}"
  TEMPERATURE="${TEMPERATURE:-1.0}"
  
  echo "==> Service: ${BASE_URL}"
  echo "==> Model:   ${MODEL}"
  echo "==> Prompt:  ${PROMPT}"
  echo
  
  echo "==> Sending request..."
  RESPONSE_FILE="$(mktemp)"
  START_TIME="$(date +%s.%N)"
  HTTP_CODE="$(
    curl -sS -w "%{http_code}" -o "${RESPONSE_FILE}" \
      "${BASE_URL}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer dummy" \
      -d "$(cat <<EOF
  {
    "model": "${MODEL}",
    "messages": [
      {"role": "user", "content": "${PROMPT}"}
    ],
    "chat_template_kwargs": {"enable_thinking": true},
    "do_sample": true,
    "top_p": 0.95,
    "top_k": 40,
    "max_tokens": ${MAX_TOKENS},
    "temperature": ${TEMPERATURE},
    "stream": false
  }
  EOF
  )"
  )"
  END_TIME="$(date +%s.%N)"
  
  echo "==> HTTP status: ${HTTP_CODE}"
  echo
  
  if [[ "${HTTP_CODE}" != "200" ]]; then
    echo "==> Request failed. Response body:"
    cat "${RESPONSE_FILE}"
    rm -f "${RESPONSE_FILE}"
    exit 1
  fi
  
  ELAPSED_SECONDS="$(
    python - <<'PY' "${START_TIME}" "${END_TIME}"
  import sys
  
  start = float(sys.argv[1])
  end = float(sys.argv[2])
  print(f"{end - start:.3f}")
  PY
  )"
  
  echo "==> Raw response:"
  if command -v jq >/dev/null 2>&1; then
    jq . "${RESPONSE_FILE}"
  else
    cat "${RESPONSE_FILE}"
  fi
  echo
  
  echo "==> Assistant reply:"
  if command -v jq >/dev/null 2>&1; then
    jq -r '.choices[0].message.content // empty' "${RESPONSE_FILE}"
  else
    python - <<'PY' "${RESPONSE_FILE}"
  import json
  import sys
  
  with open(sys.argv[1], encoding="utf-8") as f:
      data = json.load(f)
  
  message = data.get("choices", [{}])[0].get("message", {})
  content = message.get("content")
  reasoning = message.get("reasoning_content")
  
  if reasoning:
      print("--- reasoning ---")
      print(reasoning)
      print("--- content ---")
  if content:
      print(content)
  else:
      print(json.dumps(data, ensure_ascii=False, indent=2))
  PY
  fi
  
  echo
  echo "==> Throughput:"
  python - <<'PY' "${RESPONSE_FILE}" "${ELAPSED_SECONDS}"
  import json
  import sys
  
  response_file = sys.argv[1]
  elapsed_seconds = float(sys.argv[2])
  
  with open(response_file, encoding="utf-8") as f:
      data = json.load(f)
  
  usage = data.get("usage") or {}
  prompt_tokens = usage.get("prompt_tokens")
  completion_tokens = usage.get("completion_tokens")
  total_tokens = usage.get("total_tokens")
  
  def throughput(tokens):
      if tokens is None or elapsed_seconds <= 0:
          return "N/A"
      return f"{tokens / elapsed_seconds:.2f} tokens/s"
  
  print(f"Elapsed: {elapsed_seconds:.3f} s")
  print(f"Prompt tokens: {prompt_tokens if prompt_tokens is not None else 'N/A'}")
  print(f"Completion tokens: {completion_tokens if completion_tokens is not None else 'N/A'}")
  print(f"Total tokens: {total_tokens if total_tokens is not None else 'N/A'}")
  print(f"Completion throughput: {throughput(completion_tokens)}")
  print(f"Total throughput: {throughput(total_tokens)}")
  PY
  
  rm -f "${RESPONSE_FILE}"
  echo
  echo "==> Done."
  ```

  预期返回答案C。

- Single image

  Replace `${IMAGE_PATH}` with a local image path on the client side.
  Start the service with image input enabled, for example `--limit-mm-per-prompt '{"image":1}'`.

  ```bash
  IMAGE_PATH=/path/to/image.jpg
  IMAGE_BASE64="$(base64 -w 0 "${IMAGE_PATH}")"

  curl http://127.0.0.1:11223/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"minimax-m3\",
      \"messages\": [
        {
          \"role\": \"user\",
          \"content\": [
            {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,${IMAGE_BASE64}\"}},
            {\"type\": \"text\", \"text\": \"请简要描述这张图片。\"}
          ]
        }
      ],
      \"max_tokens\": 512,
      \"temperature\": 0
    }"
  ```

- Multiple images

  Start the service with a matching image limit. For the following two-image request, use `--limit-mm-per-prompt '{"image":2}'`.

  ```bash
  IMAGE1_BASE64="$(base64 -w 0 /path/to/image1.jpg)"
  IMAGE2_BASE64="$(base64 -w 0 /path/to/image2.jpg)"

  curl http://127.0.0.1:11223/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"minimax-m3\",
      \"messages\": [
        {
          \"role\": \"user\",
          \"content\": [
            {\"type\": \"text\", \"text\": \"请按顺序分别描述这两张图片，并说明它们是否相同。\"},
            {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,${IMAGE1_BASE64}\"}},
            {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,${IMAGE2_BASE64}\"}}
          ]
        }
      ],
      \"max_tokens\": 512,
      \"temperature\": 0
    }"
  ```

- Single video

  Start the service with video input enabled, for example `--limit-mm-per-prompt '{"video":1}'`.

  vLLM defaults to sampling 32 frames for video input. For regular functional verification, it is recommended to specify a smaller frame count with `media_io_kwargs`, for example 8 or 16 frames.

  ```bash
  curl http://127.0.0.1:11223/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "minimax-m3",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "video_url",
              "video_url": {
                "url": "file:///path/to/video.mp4"
              }
            },
            {
              "type": "text",
              "text": "请简要描述这个视频的主要内容。"
            }
          ]
        }
      ],
      "media_io_kwargs": {
        "video": {
          "num_frames": 8
        }
      },
      "max_tokens": 512,
      "temperature": 0
    }'
  ```

- Image and video mixed request

  Start the service with both image and video input enabled. For the following request, use `--limit-mm-per-prompt '{"image":1,"video":1}'`.

  ```bash
  IMAGE_BASE64="$(base64 -w 0 /path/to/image.jpg)"

  curl http://127.0.0.1:11223/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"minimax-m3\",
      \"messages\": [
        {
          \"role\": \"user\",
          \"content\": [
            {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,${IMAGE_BASE64}\"}},
            {\"type\": \"video_url\", \"video_url\": {\"url\": \"file:///path/to/video.mp4\"}},
            {\"type\": \"text\", \"text\": \"请分别描述图片和视频，并说明二者是否有关联。\"}
          ]
        }
      ],
      \"media_io_kwargs\": {
        \"video\": {
          \"num_frames\": 8
        }
      },
      \"max_tokens\": 512,
      \"temperature\": 0
    }"
  ```

## Supported Feature

| Model                         | Supported Hardware | BF16 | W8A8 | MXFP8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallel | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | Thinking | Tool Call | Image | Video | Max Model Len |
|-------------------------------|------|--------------------|------|------|-----------------|------------------------|------|----------------------|------------------|-----------------|-------------------|-----------------|---------------|-------------------------------|--------------------|--------------------|---------------|---------------|---------------|---------------|---------------|
| Minimax m3 | A2/A3 | ✅ | ✅ | ✖️ | ✅ | ✅ | - | - | ✅ | ✅ | - | ✅ | ✅ | ✖️ | ✖️ | ✅ | ✅ | ✖️ | ✅ | ✅ | 1M |

* 请参阅 [特性指南](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_features.html) 获取特性配置说明。
* 模型支持最长上下文长度为1M，A3单机BF16权重实测可达42K，A3单机W8A8权重实测可达128K。

## Precision
### 使用 AISBench

详细步骤请参阅 [使用 AISBench 进行性能评估](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/evaluation/using_ais_bench.html#execute-performance-evaluation)。

| Dataset | Hardward | Score | max-model-len | max-num-seqs | max_out_len | batch_size | generation_kwargs |
|---------|----------|-------|---------------|--------------|-------------|------------|-------------------|
| GSM8K   | GPU      | 96.72 | 65536         | 16           | 49152       | 16         | temperature=1.0, top_p=0.95 |
| GSM8K   | NPU      | 96.36 | 10240         | 16           | 9500        | 20         | temperature=1.0, top_p=0.95 |
| AIME2025 | GPU     | 95@repeat4 | -        | -            | -           | -          | -                 |
| AIME2025 | NPU     | 90    | -             | -            | -           | -          | temperature=1.0, top_p=0.95 |

## FAQ
- Q: 重装vLLM Ascend

  A: 可以使用如下命令重装vLLM Ascend，并直接使用当前 Python 环境里的依赖来构建
  ```
  pip install -v --no-build-isolation -e . -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
  ```

- Q: 遇到算子包安装失败的问题?
  
  A: 由于Minimax-m3适配代码未引入自定义Ascend C算子，所以可以跳过Ascend C算子编译，`export COMPILE_CUSTOM_KERNELS=0`

- Q: 遇到`TypeError: _LazyConfigMapping.__init__() missing 1 required positional argument: 'mapping'`报错 （详细报错信息如下）
  
  ```
  Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/spawn.py", line 122, in spawn_main
    exitcode = _main(fd, parent_sentinel)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/python3.12.13/lib/python3.12/multiprocessing/spawn.py", line 132, in _main
    self = reduction.pickle.load(from_parent)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  TypeError: _LazyConfigMapping.__init__() missing 1 required positional argument: 'mapping'
  ```
  
  A: 修改权重中的`configuration_minimax_m3_vl.py`文件，将`from transformers.models.auto import CONFIG_MAPPING`注释并移动到调用点中加载
  ```
  from transformers.configuration_utils import PretrainedConfig
  # from transformers.models.auto import CONFIG_MAPPING
  
  
  def _coerce_sub_config(
      sub_config: Optional[dict], default_model_type: str
  ) -> Optional[PretrainedConfig]:
      """Convert a config dict to a ``PretrainedConfig`` instance.
  
      If ``model_type`` is registered in HF ``CONFIG_MAPPING`` the corresponding
      config class is used; otherwise we fall back to a generic
      ``PretrainedConfig`` so all dict keys still become real attributes (M3's
      text backbone uses ``model_type="minimax_m2"`` which is not in
      ``CONFIG_MAPPING``).
      """
      if not isinstance(sub_config, dict):
          return sub_config
      model_type = sub_config.get("model_type", default_model_type)
      from transformers.models.auto import CONFIG_MAPPING
      cls = CONFIG_MAPPING.get(model_type, PretrainedConfig)
      return cls(**sub_config)
  ```

- Q: 发送视频请求时，没有指定 `media_io_kwargs.video.num_frames`，请求耗时较长或触发执行超时，怎么办？

  A: vLLM 的视频读取默认抽取 32 帧。MiniMax-M3 每帧会产生较多视觉 token，32 帧视频会显著增加 prefill 计算量。普通功能验证建议显式指定 8 或 16 帧：

  ```json
  {
    "media_io_kwargs": {
      "video": {
        "num_frames": 8
      }
    }
  }
  ```

  在 vLLM 0.23.0 + vLLM Ascend main 同步后的代码路径下，默认 32 帧请求已验证可在 `FULL_AND_PIECEWISE` graph 模式跑通：

  ```bash
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8]}'
  ```

  如果使用旧版本代码或仍遇到 32 帧长 prefill 超时，可使用 eager 模式和更小的 batch token 作为排障/兼容配置：

  ```bash
  export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
  export VLLM_ENGINE_ITERATION_TIMEOUT_S=1200

  vllm serve ${WEIGHT_PATH} \
    ... \
    --enforce-eager \
    --max-num-batched-tokens 1024
  ```

  如果日志中反复出现 `No available shared memory broadcast block found in 60 seconds`，通常表示 EngineCore 正在等待 worker 完成长耗时任务，例如图编译、权重量化、KV cache 量化或大 prefill 执行；它不是视频解码阶段的直接报错。

## 声明
1）当前仅为尝鲜体验，性能优化中。<br>
2）该补丁仅用于功能体验，多模态功能未充分验证，不建议直接用于生产环境。<br>
3）本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。<br>
