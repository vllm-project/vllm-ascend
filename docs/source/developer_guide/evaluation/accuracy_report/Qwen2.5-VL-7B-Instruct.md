# Qwen2.5-VL-7B-Instruct
  <div>
    <strong>vLLM version:</strong> vLLM: v0.7.3, vLLM Ascend: v0.7.3 <br>
  </div>
  <div>
      <strong>Software Environment:</strong> CANN: 8.1.0, PyTorch: 2.5.1, torch-npu: 2.5.1 <br>
  </div>
  <div>
      <strong>Hardware Environment</strong>: Atlas A2 Series <br>
  </div>
  <div>
      <strong>Datasets</strong>: mmmu_val <br>
  </div>
  <div>
      <strong>Command</strong>:

  ```bash
  export MODEL_AEGS='Qwen/Qwen2.5-VL-7B-Instruct, max_model_len=8192,dtype=auto,tensor_parallel_size=2,max_images=2'
lm_eval --model vllm-vlm --modlel_args $MODEL_ARGS --tasks mmmu_val \
--apply_chat_template --fewshot_as_multiturn  --batch_size 1
  ```
  </div>
  <div>&nbsp;</div>

| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| mmmu_val                              | none   | 0      | acc    | ↑ 0.5122 | ± 0.0162 |
<details>
<summary>mmmu_val details</summary>

| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| mmmu_val                              | none   | 0      | acc    | ↑ 0.5122 | ± 0.0162 |
| - Art and Design                      | none   | 0      | acc    | ↑ 0.6667 | ± 0.0424 |
| - Art                                 | none   | 0      | acc    | ↑ 0.6667 | ± 0.0875 |
| - Art Theory                          | none   | 0      | acc    | ↑ 0.8333 | ± 0.0692 |
| - Design                              | none   | 0      | acc    | ↑ 0.6667 | ± 0.0875 |
| - Music                               | none   | 0      | acc    | ↑ 0.5000 | ± 0.0928 |
| - Business                            | none   | 0      | acc    | ↑ 0.4133 | ± 0.0404 |
| - Accounting                          | none   | 0      | acc    | ↑ 0.4333 | ± 0.0920 |
| - Economics                           | none   | 0      | acc    | ↑ 0.5333 | ± 0.0926 |
| - Finance                             | none   | 0      | acc    | ↑ 0.3333 | ± 0.0875 |
| - Manage                              | none   | 0      | acc    | ↑ 0.3333 | ± 0.0875 |
| - Marketing                           | none   | 0      | acc    | ↑ 0.4333 | ± 0.0920 |
| - Health and Medicine                 | none   | 0      | acc    | ↑ 0.5867 | ± 0.0406 |
| - Basic Medical Science               | none   | 0      | acc    | ↑ 0.6000 | ± 0.0910 |
| - Clinical Medicine                   | none   | 0      | acc    | ↑ 0.6333 | ± 0.0895 |
| - Diagnostics and Laboratory Medicine | none   | 0      | acc    | ↑ 0.4667 | ± 0.0926 |
| - Pharmacy                            | none   | 0      | acc    | ↑ 0.6000 | ± 0.0910 |
| - Public Health                       | none   | 0      | acc    | ↑ 0.6333 | ± 0.0895 |
| - Humanities and Social Science       | none   | 0      | acc    | ↑ 0.7000 | ± 0.0413 |
| - History                             | none   | 0      | acc    | ↑ 0.7000 | ± 0.0851 |
| - Literature                          | none   | 0      | acc    | ↑ 0.8333 | ± 0.0692 |
| - Psychology                          | none   | 0      | acc    | ↑ 0.7333 | ± 0.0821 |
| - Sociology                           | none   | 0      | acc    | ↑ 0.5333 | ± 0.0926 |
| - Science                             | none   | 0      | acc    | ↑ 0.4200 | ± 0.0409 |
| - Biology                             | none   | 0      | acc    | ↑ 0.4000 | ± 0.0910 |
| - Chemistry                           | none   | 0      | acc    | ↑ 0.3667 | ± 0.0895 |
| - Geography                           | none   | 0      | acc    | ↑ 0.4667 | ± 0.0926 |
| - Math                                | none   | 0      | acc    | ↑ 0.4333 | ± 0.0920 |
| - Physics                             | none   | 0      | acc    | ↑ 0.4333 | ± 0.0920 |
| - Tech and Engineering                | none   | 0      | acc    | ↑ 0.4000 | ± 0.0337 |
| - Agriculture                         | none   | 0      | acc    | ↑ 0.5333 | ± 0.0926 |
| - Architecture and Engineering        | none   | 0      | acc    | ↑ 0.4333 | ± 0.0920 |
| - Computer Science                    | none   | 0      | acc    | ↑ 0.4000 | ± 0.0910 |
| - Electronics                         | none   | 0      | acc    | ↑ 0.2667 | ± 0.0821 |
| - Energy and Power                    | none   | 0      | acc    | ↑ 0.2667 | ± 0.0821 |
| - Materials                           | none   | 0      | acc    | ↑ 0.4000 | ± 0.0910 |
| - Mechanical Engineering              | none   | 0      | acc    | ↑ 0.5000 | ± 0.0928 |
</details>
