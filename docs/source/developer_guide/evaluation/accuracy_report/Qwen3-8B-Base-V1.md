# 🎯 Qwen3-8B-Base Accuracy Test
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
      <strong>Datasets</strong>: ceval-valid,gsm8k <br>
  </div>
  <div>
      <strong>Command</strong>: 

  ```bash
  export MODEL_ARGS='pretrained=Qwen/Qwen3-8B-Base, max_model_len=4096,dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.6'
lm_eval --model vllm --model_args $MODEL_ARGS --tasks ceval-valid,gsm8k \ 
--apply_chat_template --fewshot_as_multiturn --num_fewshot 5 --batch_size 1
  ```
  </div>
  <div>&nbsp;</div>
  
| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| ceval-valid                           | none   | 5      | acc    | ↑ 0.2303 | ± 0.0115 |
| gsm8k                                 | flexible-extract | 5      | exact_match | ↑ 0.8294 | ± 0.0104 |
<details>
<summary>ceval-valid details</summary>

| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| ceval-valid                           | none   | 5      | acc    | ↑ 0.2303 | ± 0.0115 |
| - ceval-valid_accountant              | none   | 5      | acc    | ↑ 0.2245 | ± 0.0602 |
| - ceval-valid_advanced_mathematics    | none   | 5      | acc    | ↑ 0.3158 | ± 0.1096 |
| - ceval-valid_art_studies             | none   | 5      | acc    | ↑ 0.4545 | ± 0.0880 |
| - ceval-valid_basic_medicine          | none   | 5      | acc    | ↑ 0.0526 | ± 0.0526 |
| - ceval-valid_business_administration | none   | 5      | acc    | ↑ 0.2424 | ± 0.0758 |
| - ceval-valid_chinese_language_and_literature | none   | 5      | acc    | ↑ 0.2174 | ± 0.0879 |
| - ceval-valid_civil_servant           | none   | 5      | acc    | ↑ 0.2553 | ± 0.0643 |
| - ceval-valid_clinical_medicine       | none   | 5      | acc    | ↑ 0.2273 | ± 0.0914 |
| - ceval-valid_college_chemistry       | none   | 5      | acc    | ↑ 0.1667 | ± 0.0777 |
| - ceval-valid_college_economics       | none   | 5      | acc    | ↑ 0.2909 | ± 0.0618 |
| - ceval-valid_college_physics         | none   | 5      | acc    | ↑ 0.2105 | ± 0.0961 |
| - ceval-valid_college_programming     | none   | 5      | acc    | ↑ 0.2432 | ± 0.0715 |
| - ceval-valid_computer_architecture   | none   | 5      | acc    | ↑ 0.2857 | ± 0.1010 |
| - ceval-valid_computer_network        | none   | 5      | acc    | ↑ 0.1053 | ± 0.0723 |
| - ceval-valid_discrete_mathematics    | none   | 5      | acc    | ↑ 0.3750 | ± 0.1250 |
| - ceval-valid_education_science       | none   | 5      | acc    | ↑ 0.2414 | ± 0.0809 |
| - ceval-valid_electrical_engineer     | none   | 5      | acc    | ↑ 0.2162 | ± 0.0686 |
| - ceval-valid_environmental_impact_assessment_engineer | none   | 5      | acc    | ↑ 0.1613 | ± 0.0672 |
| - ceval-valid_fire_engineer           | none   | 5      | acc    | ↑ 0.2581 | ± 0.0799 |
| - ceval-valid_high_school_biology     | none   | 5      | acc    | ↑ 0.3684 | ± 0.1137 |
| - ceval-valid_high_school_chemistry   | none   | 5      | acc    | ↑ 0.2105 | ± 0.0961 |
| - ceval-valid_high_school_chinese     | none   | 5      | acc    | ↑ 0.2105 | ± 0.0961 |
| - ceval-valid_high_school_geography   | none   | 5      | acc    | ↑ 0.2105 | ± 0.0961 |
| - ceval-valid_high_school_history     | none   | 5      | acc    | ↑ 0.3000 | ± 0.1051 |
| - ceval-valid_high_school_mathematics | none   | 5      | acc    | ↑ 0.2222 | ± 0.1008 |
| - ceval-valid_high_school_physics     | none   | 5      | acc    | ↑ 0.2105 | ± 0.0961 |
| - ceval-valid_high_school_politics    | none   | 5      | acc    | ↑ 0.2105 | ± 0.0961 |
| - ceval-valid_ideological_and_moral_cultivation | none   | 5      | acc    | ↑ 0.2632 | ± 0.1038 |
| - ceval-valid_law                     | none   | 5      | acc    | ↑ 0.2083 | ± 0.0847 |
| - ceval-valid_legal_professional      | none   | 5      | acc    | ↑ 0.0435 | ± 0.0435 |
| - ceval-valid_logic                   | none   | 5      | acc    | ↑ 0.1818 | ± 0.0842 |
| - ceval-valid_mao_zedong_thought      | none   | 5      | acc    | ↑ 0.3333 | ± 0.0983 |
| - ceval-valid_marxism                 | none   | 5      | acc    | ↑ 0.2632 | ± 0.1038 |
| - ceval-valid_metrology_engineer      | none   | 5      | acc    | ↑ 0.1250 | ± 0.0690 |
| - ceval-valid_middle_school_biology   | none   | 5      | acc    | ↑ 0.1905 | ± 0.0878 |
| - ceval-valid_middle_school_chemistry | none   | 5      | acc    | ↑ 0.1500 | ± 0.0819 |
| - ceval-valid_middle_school_geography | none   | 5      | acc    | ↑ 0.0833 | ± 0.0833 |
| - ceval-valid_middle_school_history   | none   | 5      | acc    | ↑ 0.1818 | ± 0.0842 |
| - ceval-valid_middle_school_mathematics | none   | 5      | acc    | ↑ 0.1579 | ± 0.0859 |
| - ceval-valid_middle_school_physics   | none   | 5      | acc    | ↑ 0.2105 | ± 0.0961 |
| - ceval-valid_middle_school_politics  | none   | 5      | acc    | ↑ 0.2857 | ± 0.1010 |
| - ceval-valid_modern_chinese_history  | none   | 5      | acc    | ↑ 0.1739 | ± 0.0808 |
| - ceval-valid_operating_system        | none   | 5      | acc    | ↑ 0.1579 | ± 0.0859 |
| - ceval-valid_physician               | none   | 5      | acc    | ↑ 0.2653 | ± 0.0637 |
| - ceval-valid_plant_protection        | none   | 5      | acc    | ↑ 0.3182 | ± 0.1016 |
| - ceval-valid_probability_and_statistics | none   | 5      | acc    | ↑ 0.1111 | ± 0.0762 |
| - ceval-valid_professional_tour_guide | none   | 5      | acc    | ↑ 0.3448 | ± 0.0898 |
| - ceval-valid_sports_science          | none   | 5      | acc    | ↑ 0.1053 | ± 0.0723 |
| - ceval-valid_tax_accountant          | none   | 5      | acc    | ↑ 0.2041 | ± 0.0582 |
| - ceval-valid_teacher_qualification   | none   | 5      | acc    | ↑ 0.2955 | ± 0.0696 |
| - ceval-valid_urban_and_rural_planner | none   | 5      | acc    | ↑ 0.2174 | ± 0.0615 |
| - ceval-valid_veterinary_medicine     | none   | 5      | acc    | ↑ 0.2174 | ± 0.0879 |
</details>
<details>
<summary>gsm8k details</summary>

| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| gsm8k                                 | flexible-extract | 5      | exact_match | ↑ 0.8294 | ± 0.0104 |
</details>
