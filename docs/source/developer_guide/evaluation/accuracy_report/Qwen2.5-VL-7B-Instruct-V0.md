# 🎯 Qwen2.5-VL-7B-Instruct Accuracy Test
  <div>
    <strong>vLLM version:</strong> vLLM: 0.9.1, vLLM Ascend: main <br>
  </div>
  <div>
      <strong>Software Environment:</strong> CANN: 8.1.RC1, PyTorch: 2.5.1, torch-npu: 2.5.1 <br>
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
  export MODEL_ARGS='pretrained=Qwen/Qwen2.5-VL-7B-Instruct, max_model_len=8192,dtype=auto,tensor_parallel_size=4,max_images=2'
lm_eval --model vllm-vlm --model_args $MODEL_ARGS --tasks mmmu_val \ 
--apply_chat_template --fewshot_as_multiturn  --batch_size 1
  ```
  </div>
  <div>&nbsp;</div>
  
| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| mmmu_val                              | none   | 0      | acc    | ↑ 0.5178 | ± 0.0163 |
<details>
<summary>mmmu_val details</summary>

| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| mmmu_val                              | none   | 0      | acc    | ↑ 0.5178 | ± 0.0163 |
| - Art and Design                      | none   | 0      | acc    | ↑ 0.6583 | ± 0.0424 |
| - Art                                 | none   | 0      | acc    | ↑ 0.6667 | ± 0.0875 |
| - Art Theory                          | none   | 0      | acc    | ↑ 0.8333 | ± 0.0692 |
| - Design                              | none   | 0      | acc    | ↑ 0.6667 | ± 0.0875 |
| - Music                               | none   | 0      | acc    | ↑ 0.4667 | ± 0.0926 |
| - Business                            | none   | 0      | acc    | ↑ 0.4333 | ± 0.0407 |
| - Accounting                          | none   | 0      | acc    | ↑ 0.4333 | ± 0.0920 |
| - Economics                           | none   | 0      | acc    | ↑ 0.5333 | ± 0.0926 |
| - Finance                             | none   | 0      | acc    | ↑ 0.3667 | ± 0.0895 |
| - Manage                              | none   | 0      | acc    | ↑ 0.3333 | ± 0.0875 |
| - Marketing                           | none   | 0      | acc    | ↑ 0.5000 | ± 0.0928 |
| - Health and Medicine                 | none   | 0      | acc    | ↑ 0.5800 | ± 0.0405 |
| - Basic Medical Science               | none   | 0      | acc    | ↑ 0.6000 | ± 0.0910 |
| - Clinical Medicine                   | none   | 0      | acc    | ↑ 0.6000 | ± 0.0910 |
| - Diagnostics and Laboratory Medicine | none   | 0      | acc    | ↑ 0.4333 | ± 0.0920 |
| - Pharmacy                            | none   | 0      | acc    | ↑ 0.6333 | ± 0.0895 |
| - Public Health                       | none   | 0      | acc    | ↑ 0.6333 | ± 0.0895 |
| - Humanities and Social Science       | none   | 0      | acc    | ↑ 0.6917 | ± 0.0419 |
| - History                             | none   | 0      | acc    | ↑ 0.7000 | ± 0.0851 |
| - Literature                          | none   | 0      | acc    | ↑ 0.8000 | ± 0.0743 |
| - Psychology                          | none   | 0      | acc    | ↑ 0.7333 | ± 0.0821 |
| - Sociology                           | none   | 0      | acc    | ↑ 0.5333 | ± 0.0926 |
| - Science                             | none   | 0      | acc    | ↑ 0.4067 | ± 0.0407 |
| - Biology                             | none   | 0      | acc    | ↑ 0.3667 | ± 0.0895 |
| - Chemistry                           | none   | 0      | acc    | ↑ 0.3667 | ± 0.0895 |
| - Geography                           | none   | 0      | acc    | ↑ 0.4667 | ± 0.0926 |
| - Math                                | none   | 0      | acc    | ↑ 0.4333 | ± 0.0920 |
| - Physics                             | none   | 0      | acc    | ↑ 0.4000 | ± 0.0910 |
| - Tech and Engineering                | none   | 0      | acc    | ↑ 0.4333 | ± 0.0342 |
| - Agriculture                         | none   | 0      | acc    | ↑ 0.5333 | ± 0.0926 |
| - Architecture and Engineering        | none   | 0      | acc    | ↑ 0.4667 | ± 0.0926 |
| - Computer Science                    | none   | 0      | acc    | ↑ 0.4333 | ± 0.0920 |
| - Electronics                         | none   | 0      | acc    | ↑ 0.3333 | ± 0.0875 |
| - Energy and Power                    | none   | 0      | acc    | ↑ 0.2667 | ± 0.0821 |
| - Materials                           | none   | 0      | acc    | ↑ 0.4667 | ± 0.0926 |
| - Mechanical Engineering              | none   | 0      | acc    | ↑ 0.5333 | ± 0.0926 |
</details>
