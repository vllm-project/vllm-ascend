# Minimax m3

## Environment Preparation

### Model Weight

`Minimax-m3`(BF16 version): requires 1 Ascend 910C (with 16 x 64G NPUs). [Download model weight](https://www.modelscope.cn/collections/MiniMax/MiniMax-M3)

It is recommended to place the model weight in a shared cache directory.

### Installation

- Step 1： Download v0.20.2rc1 Docker image
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

- Step 3: Update vLLM Ascend
  ```
  cd /vllm-workspace/vllm-ascend
  git fetch origin pull/10682/merge:pr-10682
  git merge pr-10682
  ```

## Deployment

Start the online serving service with the following command:

```
vllm serve ${WEIGHT_PATH} \
  --served-model-name minimax-m3 \
  --trust-remote-code \
  --max-model-len 46080 \
  --tensor-parallel-size 16 \
  --enable-expert-parallel \
  --max-num-seqs 8 \
  --distributed_executor_backend "mp" \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser minimax_m3 \
  --limit-mm-per-prompt '{"image":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
  --additional-config '{"enable_cpu_binding":true,"ascend_compilation_config":{"fuse_qknorm_rope":false, "fuse_norm_quant":false}}' \
  --port 11223 > ${LOG_PATH} 2>&1 & 
```

## Functional Verification

- Text

  ```
  #!/bin/bash
  # Verify MiniMax M3 vLLM service via OpenAI-compatible chat completions API.
  # Usage: bash verify_curl.sh
  
  set -euo pipefail
  
  BASE_URL="${BASE_URL:-http://127.0.0.1:11223}"
  MODEL="${MODEL:-minimax-m3}"
  PROMPT="Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n\nA student regrets that he fell asleep during a lecture in electrochemistry, facing the following incomplete statement in a test:\nThermodynamically, oxygen is a …… oxidant in basic solutions. Kinetically, oxygen reacts …… in acidic solutions.\nWhich combination of weaker/stronger and faster/slower is correct?\n\nA) weaker – faster\nB) stronger – faster\nC) weaker - slower\nD) stronger – slower"
  MAX_TOKENS="${MAX_TOKENS:-5120}"
  TEMPERATURE="${TEMPERATURE:-1.0}"
  
  echo "==> Service: ${BASE_URL}"
  echo "==> Model:   ${MODEL}"
  echo "==> Prompt:  ${PROMPT}"
  echo
  
  echo "==> Sending request..."
  RESPONSE_FILE="$(mktemp)"
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
  
  echo "==> HTTP status: ${HTTP_CODE}"
  echo
  
  if [[ "${HTTP_CODE}" != "200" ]]; then
    echo "==> Request failed. Response body:"
    cat "${RESPONSE_FILE}"
    rm -f "${RESPONSE_FILE}"
    exit 1
  fi
  
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
  
  rm -f "${RESPONSE_FILE}"
  echo
  echo "==> Done."
  ```

  预期返回答案C。

## Supported Feature

| Model                         | Supported Hardware | BF16 | W8A8 | MXFP8 | Chunked Prefill | Automatic Prefix Cache | LoRA | Speculative Decoding | Async Scheduling | Tensor Parallel | Pipeline Parallel | Expert Parallel | Data Parallel | Prefill-decode Disaggregation | Piecewise AclGraph | Fullgraph AclGraph | Thinking | Tool Call | Image | Video | Max Model Len |
|-------------------------------|------|--------------------|------|------|-----------------|------------------------|------|----------------------|------------------|-----------------|-------------------|-----------------|---------------|-------------------------------|--------------------|--------------------|---------------|---------------|---------------|---------------|---------------|
| Minimax m3 | A2/A3 | ✅ | ✅ | ✖️ | ✅ | ✅ | - | - | ✅ | ✅ | - | ✅ | ✅ | ✖️ | ✖️ | ✅ | ✅ | ✖️ | 单图 | ✖️ | 1M |

* 请参阅 [特性指南](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/support_matrix/supported_features.html) 获取特性配置说明。
* 模型支持最长上下文长度为1M，A3单机BF16权重实测可达45K。

## Precision
### 使用 AISBench

详细步骤请参阅 [使用 AISBench 进行性能评估](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/evaluation/using_ais_bench.html#execute-performance-evaluation)。

| Dataset | Hardward | Score | max-model-len | max-num-seqs | max_out_len | batch_size | generation_kwargs |
|---------|----------|-------|---------------|--------------|-------------|------------|-------------------|
| GSM8K   | GPU      | 96.72 | 65536         | 16           | 49152       | 16         | temperature=1.0, top_p=0.95 |
| GSM8K   | NPU      | 96.36 | 10240         | 16           | 9500        | 20         | temperature=1.0, top_p=0.95 |

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

## 声明
1）当前仅为尝鲜体验，性能优化中。<br>
2）该补丁仅用于功能体验，多模态功能未充分验证，不建议直接用于生产环境。<br>
3）本代码仓提到的数据集和模型仅作为示例，这些数据集和模型仅供您用于非商业目的，如您使用这些数据集和模型来完成示例，请您特别注意应遵守对应数据集和模型的License，如您因使用数据集或模型而产生侵权纠纷，华为不承担任何责任。<br>
