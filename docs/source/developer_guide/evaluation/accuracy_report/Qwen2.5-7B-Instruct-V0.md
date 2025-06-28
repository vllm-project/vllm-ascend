# 🎯 Qwen2.5-7B-Instruct
**vLLM Version**: vLLM: 0.9.1 ([b6553be](https://github.com/vllm-project/vllm/commit/b6553be1bc75f046b00046a4ad7576364d03c835)), **vLLM Ascend**: main ([b308a7a](https://github.com/vllm-project/vllm-ascend/commit/b308a7a25897b88d4a23a9e3d583f4ec6de256ac))
**vLLM Engine**: V0  
**Software Environment**: CANN: 8.1.RC1, PyTorch: 2.5.1, torch-npu: 2.5.1.post1.dev20250619  
**Hardware Environment**: Atlas A2 Series  
**Datasets**: ceval-valid,gsm8k  
**Command**:  
```bash
export MODEL_ARGS='pretrained=Qwen/Qwen2.5-7B-Instruct,max_model_len=4096,dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.6'
lm_eval --model vllm --model_args $MODEL_ARGS --tasks ceval-valid,gsm8k \ 
--apply_chat_template --fewshot_as_multiturn --num_fewshot 5 --batch_size 1
```
  
| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| ceval-valid                           | none   | 5      | acc_norm | ✅0.8001 | ± 0.0105 |
| gsm8k                                 | flexible-extract | 5      | exact_match | ✅0.7278 | ± 0.0123 |
<details>
<summary>ceval-valid details</summary>

| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| ceval-valid                           | none   | 5      | acc_norm | ✅0.8001 | ± 0.0105 |
| - ceval-valid_accountant              | none   | 5      | acc    | 0.8776 | ± 0.0473 |
| - ceval-valid_advanced_mathematics    | none   | 5      | acc    | 0.4211 | ± 0.1164 |
| - ceval-valid_art_studies             | none   | 5      | acc    | 0.7273 | ± 0.0787 |
| - ceval-valid_basic_medicine          | none   | 5      | acc    | 0.9474 | ± 0.0526 |
| - ceval-valid_business_administration | none   | 5      | acc    | 0.8485 | ± 0.0634 |
| - ceval-valid_chinese_language_and_literature | none   | 5      | acc    | 0.6087 | ± 0.1041 |
| - ceval-valid_civil_servant           | none   | 5      | acc    | 0.8298 | ± 0.0554 |
| - ceval-valid_clinical_medicine       | none   | 5      | acc    | 0.7727 | ± 0.0914 |
| - ceval-valid_college_chemistry       | none   | 5      | acc    | 0.6250 | ± 0.1009 |
| - ceval-valid_college_economics       | none   | 5      | acc    | 0.7455 | ± 0.0593 |
| - ceval-valid_college_physics         | none   | 5      | acc    | 0.7368 | ± 0.1038 |
| - ceval-valid_college_programming     | none   | 5      | acc    | 0.8649 | ± 0.0570 |
| - ceval-valid_computer_architecture   | none   | 5      | acc    | 0.7143 | ± 0.1010 |
| - ceval-valid_computer_network        | none   | 5      | acc    | 0.6842 | ± 0.1096 |
| - ceval-valid_discrete_mathematics    | none   | 5      | acc    | 0.2500 | ± 0.1118 |
| - ceval-valid_education_science       | none   | 5      | acc    | 0.8621 | ± 0.0652 |
| - ceval-valid_electrical_engineer     | none   | 5      | acc    | 0.6757 | ± 0.0780 |
| - ceval-valid_environmental_impact_assessment_engineer | none   | 5      | acc    | 0.7419 | ± 0.0799 |
| - ceval-valid_fire_engineer           | none   | 5      | acc    | 0.7419 | ± 0.0799 |
| - ceval-valid_high_school_biology     | none   | 5      | acc    | 0.8947 | ± 0.0723 |
| - ceval-valid_high_school_chemistry   | none   | 5      | acc    | 0.7368 | ± 0.1038 |
| - ceval-valid_high_school_chinese     | none   | 5      | acc    | 0.6842 | ± 0.1096 |
| - ceval-valid_high_school_geography   | none   | 5      | acc    | 0.8947 | ± 0.0723 |
| - ceval-valid_high_school_history     | none   | 5      | acc    | 0.9000 | ± 0.0688 |
| - ceval-valid_high_school_mathematics | none   | 5      | acc    | 0.5000 | ± 0.1213 |
| - ceval-valid_high_school_physics     | none   | 5      | acc    | 0.7368 | ± 0.1038 |
| - ceval-valid_high_school_politics    | none   | 5      | acc    | 0.8947 | ± 0.0723 |
| - ceval-valid_ideological_and_moral_cultivation | none   | 5      | acc    | 0.9474 | ± 0.0526 |
| - ceval-valid_law                     | none   | 5      | acc    | 0.6667 | ± 0.0983 |
| - ceval-valid_legal_professional      | none   | 5      | acc    | 0.7391 | ± 0.0936 |
| - ceval-valid_logic                   | none   | 5      | acc    | 0.6364 | ± 0.1050 |
| - ceval-valid_mao_zedong_thought      | none   | 5      | acc    | 0.9583 | ± 0.0417 |
| - ceval-valid_marxism                 | none   | 5      | acc    | 0.9474 | ± 0.0526 |
| - ceval-valid_metrology_engineer      | none   | 5      | acc    | 0.8333 | ± 0.0777 |
| - ceval-valid_middle_school_biology   | none   | 5      | acc    | 0.9524 | ± 0.0476 |
| - ceval-valid_middle_school_chemistry | none   | 5      | acc    | 0.9500 | ± 0.0500 |
| - ceval-valid_middle_school_geography | none   | 5      | acc    | 0.9167 | ± 0.0833 |
| - ceval-valid_middle_school_history   | none   | 5      | acc    | 0.9091 | ± 0.0627 |
| - ceval-valid_middle_school_mathematics | none   | 5      | acc    | 0.6842 | ± 0.1096 |
| - ceval-valid_middle_school_physics   | none   | 5      | acc    | 0.9474 | ± 0.0526 |
| - ceval-valid_middle_school_politics  | none   | 5      | acc    | 1.0000 | ± 0.0000 |
| - ceval-valid_modern_chinese_history  | none   | 5      | acc    | 0.9130 | ± 0.0601 |
| - ceval-valid_operating_system        | none   | 5      | acc    | 0.8421 | ± 0.0859 |
| - ceval-valid_physician               | none   | 5      | acc    | 0.8367 | ± 0.0533 |
| - ceval-valid_plant_protection        | none   | 5      | acc    | 0.8636 | ± 0.0749 |
| - ceval-valid_probability_and_statistics | none   | 5      | acc    | 0.5556 | ± 0.1205 |
| - ceval-valid_professional_tour_guide | none   | 5      | acc    | 0.8966 | ± 0.0576 |
| - ceval-valid_sports_science          | none   | 5      | acc    | 0.9474 | ± 0.0526 |
| - ceval-valid_tax_accountant          | none   | 5      | acc    | 0.8571 | ± 0.0505 |
| - ceval-valid_teacher_qualification   | none   | 5      | acc    | 0.9091 | ± 0.0438 |
| - ceval-valid_urban_and_rural_planner | none   | 5      | acc    | 0.8043 | ± 0.0591 |
| - ceval-valid_veterinary_medicine     | none   | 5      | acc    | 0.8261 | ± 0.0808 |
</details>
<details>
<summary>gsm8k details</summary>

| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| gsm8k                                 | flexible-extract | 5      | exact_match | ✅0.7278 | ± 0.0123 |
</details>
