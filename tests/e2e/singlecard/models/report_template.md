# {{ model_name }}

- **vLLM Version**: vLLM: {{ vllm_version }} ([{{ vllm_commit[:7] }}](https://github.com/vllm-project/vllm/commit/{{ vllm_commit }})),
- **vLLM Ascend Version**: {{ vllm_ascend_version }} ([{{ vllm_ascend_commit[:7] }}](https://github.com/vllm-project/vllm-ascend/commit/{{ vllm_ascend_commit }}))  
- **Software Environment**:
  - **CANN**: {{ cann_version }}
  - **PyTorch**: {{ torch_version }}
  - **torch-npu**: {{ torch_npu_version }}  
- **Hardware Environment**: Atlas A2 Series  
- **vLLM args**:
  - **pretrained**: {{ pretrained }}
  - **tensor_parallel_size**: {{ tensor_parallel_size }}
  - **dtype**: {{ dtype  }}
  - **trust_remote_code**: {{ trust_remote_code }}
  - **max_model_len**: {{ max_model_len }}
- **lm-eval args**:
  - **model**: {{ model_type }}
  - **model_args**: vLLM args
  - **tasks**: {{ datasets }}
  - **apply_chat_template**: {{ apply_chat_template }}
  - **fewshot_as_multiturn**: {{ fewshot_as_multiturn }}
 {% if num_fewshot is defined and num_fewshot != "N/A" %} - **num_fewshot**: {{ num_fewshot }} {% endif %}
  - **limit**: {{ limit }}
  - **batch_size**: {{ batch_size}}
- **Parallel Mode**: {{ parallel_mode }}
- **Execution Mode**: ACLGraph  

**Command**:  

```bash
export MODEL_ARGS={{ model_args }}
lm_eval --model {{ model_type }} --model_args $MODEL_ARGS --tasks {{ datasets }} \
--apply_chat_template {{ apply_chat_template }} --fewshot_as_multiturn {{ fewshot_as_multiturn }} {% if num_fewshot is defined and num_fewshot != "N/A" %} --num_fewshot {{ num_fewshot }} {% endif %} \
--limit {{ limit }} --batch_size {{ batch_size}}
```

| Task                  | Metric      | Value     | Stderr |
|-----------------------|-------------|----------:|-------:|
{% for row in rows -%}
| {{ row.task }} | {{ row.metric }} | {{ "%.4f" | format(row.value | float) }} | ± {{ "%.4f" | format(row.stderr | float) }} |
{% endfor %}
