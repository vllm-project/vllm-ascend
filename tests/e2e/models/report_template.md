# {{ model_name }}

- **vLLM Version**: vLLM: {{ vllm_version }} ([{{ vllm_commit[:7] }}](https://github.com/vllm-project/vllm/commit/{{ vllm_commit }})), **vLLM Ascend Version**: {{ vllm_ascend_version }} ([{{ vllm_ascend_commit[:7] }}](https://github.com/vllm-project/vllm-ascend/commit/{{ vllm_ascend_commit }}))  
- **Software Environment**: **CANN**: {{ cann_version }}, **PyTorch**: {{ torch_version }}, **torch-npu**: {{ torch_npu_version }}  
- **Hardware Environment**: Atlas A2 Series  
- **Parallel mode**: {{ parallel_mode }}
- **Execution mode**: ACLGraph

**Command**:  

```bash
export MODEL_ARGS={{ model_args }}
lm_eval --model {{ model_type }} --model_args $MODEL_ARGS --tasks {{ datasets }} \
{% if apply_chat_template %} --apply_chat_template {{ apply_chat_template }} {% endif %} {% if fewshot_as_multiturn %} --fewshot_as_multiturn {{ fewshot_as_multiturn }} {% endif %} {% if num_fewshot is defined and num_fewshot != "N/A" %} --num_fewshot {{ num_fewshot }} {% endif %} {% if limit is defined and limit != "N/A" %} --limit {{ limit }} {% endif %} --batch_size {{ batch_size}}
```

| Task                  | Metric      | Value     | Stderr |
|-----------------------|-------------|----------:|-------:|
{% for row in rows -%}
| {{ row.task }} | {{ row.metric }} | {{ row.value }} | ± {{ "%.4f" | format(row.stderr | float) }} |
{% endfor %}
