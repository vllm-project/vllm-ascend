# {{ model_name }}

- **vLLM Version**: vLLM: {{ vllm_version }} (commit {{ vllm_commit[:7] }}), **vLLM Ascend Version**: {{ vllm_ascend_version }} (commit {{ vllm_ascend_commit[:7] }})
- **Software Environment**: **CANN**: {{ cann_version }}, **PyTorch**: {{ torch_version }}, **torch-npu**: {{ torch_npu_version }}
- **Hardware Environment**: {{ hardware }}
- **Parallel mode**: {{ parallel_mode }}
- **Execution mode**: {{ execution_model }}

**Command**:

```bash
export MODEL_ARGS={{ model_args }}
lm_eval --model {{ model_type }} --model_args $MODEL_ARGS \
  --tasks {{ datasets }} \
{%- if apply_chat_template is defined and (apply_chat_template|string|lower in ["true", "1"]) %}
  --apply_chat_template \
{%- endif %}
{%- if fewshot_as_multiturn is defined and (fewshot_as_multiturn|string|lower in ["true", "1"]) %}
  --fewshot_as_multiturn \
{%- endif %}
{%- if num_fewshot is defined and num_fewshot != "N/A" %}
  --num_fewshot {{ num_fewshot }} \
{%- endif %}
{%- if limit is defined and limit != "N/A" %}
  --limit {{ limit }} \
{%- endif %}
  --batch_size {{ batch_size }}
```

<table>
  <thead>
    <tr>
      <th>Task</th>
      <th>Metric</th>
      <th>Value</th>
      <th>Stderr</th>
    </tr>
  </thead>
  <tbody>
  {% for row in rows -%}
    <tr>
      <td>{{ row.task }}</td>
      <td>{{ row.metric }}</td>
      <td>{{ row.value }}</td>
      <td>+- {{ "%.4f" | format(row.stderr | float) }}</td>
    </tr>
  {% endfor %}
  </tbody>
</table>
