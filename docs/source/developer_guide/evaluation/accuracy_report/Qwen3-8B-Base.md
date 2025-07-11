# Qwen3-8B-Base
**vLLM Version**: vLLM: 0.9.2 ([a5dd03c](https://github.com/vllm-project/vllm/commit/a5dd03c)), vLLM Ascend: main ([d13fb07](https://github.com/vllm-project/vllm-ascend/commit/d13fb07))  
**Software Environment**: CANN: 8.1.RC1, PyTorch: 2.5.1, torch-npu: 2.5.1.post1.dev20250619  
**Hardware Environment**: Atlas A2 Series  
**Datasets**: ceval-valid,gsm8k  
**Parallel Mode**: TP  
**Execution Mode**: ACLGraph  
**Command**:  
```bash
export MODEL_ARGS='pretrained=Qwen/Qwen3-8B-Base,max_model_len=4096,dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.6'
lm_eval --model vllm --model_args $MODEL_ARGS --tasks ceval-valid,gsm8k \ 
--apply_chat_template --fewshot_as_multiturn --num_fewshot 5 --batch_size 1
```
  
| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| ceval-valid                           | none   | 5      | acc    | ✅0.8187 | ± 0.0102 |
| gsm8k                                 | flexible-extract | 5      | exact_match | ✅0.8302 | ± 0.0103 |
<details>
<summary>ceval-valid details</summary>

| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| ceval-valid                           | none   | 5      | acc    | ✅0.8187 | ± 0.0102 |
| - ceval-valid_accountant              | none   | 5      | acc    | 0.8367 | ± 0.0533 |
| - ceval-valid_advanced_mathematics    | none   | 5      | acc    | 0.6316 | ± 0.1137 |
| - ceval-valid_art_studies             | none   | 5      | acc    | 0.8182 | ± 0.0682 |
| - ceval-valid_basic_medicine          | none   | 5      | acc    | 0.8947 | ± 0.0723 |
| - ceval-valid_business_administration | none   | 5      | acc    | 0.8485 | ± 0.0634 |
| - ceval-valid_chinese_language_and_literature | none   | 5      | acc    | 0.6522 | ± 0.1015 |
| - ceval-valid_civil_servant           | none   | 5      | acc    | 0.7660 | ± 0.0624 |
| - ceval-valid_clinical_medicine       | none   | 5      | acc    | 0.8636 | ± 0.0749 |
| - ceval-valid_college_chemistry       | none   | 5      | acc    | 0.7083 | ± 0.0948 |
| - ceval-valid_college_economics       | none   | 5      | acc    | 0.7273 | ± 0.0606 |
| - ceval-valid_college_physics         | none   | 5      | acc    | 0.8421 | ± 0.0859 |
| - ceval-valid_college_programming     | none   | 5      | acc    | 0.8649 | ± 0.0570 |
| - ceval-valid_computer_architecture   | none   | 5      | acc    | 0.8095 | ± 0.0878 |
| - ceval-valid_computer_network        | none   | 5      | acc    | 0.7368 | ± 0.1038 |
| - ceval-valid_discrete_mathematics    | none   | 5      | acc    | 0.3750 | ± 0.1250 |
| - ceval-valid_education_science       | none   | 5      | acc    | 0.9310 | ± 0.0479 |
| - ceval-valid_electrical_engineer     | none   | 5      | acc    | 0.6216 | ± 0.0808 |
| - ceval-valid_environmental_impact_assessment_engineer | none   | 5      | acc    | 0.7742 | ± 0.0763 |
| - ceval-valid_fire_engineer           | none   | 5      | acc    | 0.7419 | ± 0.0799 |
| - ceval-valid_high_school_biology     | none   | 5      | acc    | 0.8947 | ± 0.0723 |
| - ceval-valid_high_school_chemistry   | none   | 5      | acc    | 0.8421 | ± 0.0859 |
| - ceval-valid_high_school_chinese     | none   | 5      | acc    | 0.6316 | ± 0.1137 |
| - ceval-valid_high_school_geography   | none   | 5      | acc    | 0.9474 | ± 0.0526 |
| - ceval-valid_high_school_history     | none   | 5      | acc    | 0.9000 | ± 0.0688 |
| - ceval-valid_high_school_mathematics | none   | 5      | acc    | 0.6111 | ± 0.1182 |
| - ceval-valid_high_school_physics     | none   | 5      | acc    | 0.9474 | ± 0.0526 |
| - ceval-valid_high_school_politics    | none   | 5      | acc    | 0.8947 | ± 0.0723 |
| - ceval-valid_ideological_and_moral_cultivation | none   | 5      | acc    | 1.0000 | ± 0.0000 |
| - ceval-valid_law                     | none   | 5      | acc    | 0.6667 | ± 0.0983 |
| - ceval-valid_legal_professional      | none   | 5      | acc    | 0.8261 | ± 0.0808 |
| - ceval-valid_logic                   | none   | 5      | acc    | 0.7727 | ± 0.0914 |
| - ceval-valid_mao_zedong_thought      | none   | 5      | acc    | 0.9167 | ± 0.0576 |
| - ceval-valid_marxism                 | none   | 5      | acc    | 0.9474 | ± 0.0526 |
| - ceval-valid_metrology_engineer      | none   | 5      | acc    | 0.8750 | ± 0.0690 |
| - ceval-valid_middle_school_biology   | none   | 5      | acc    | 0.8571 | ± 0.0782 |
| - ceval-valid_middle_school_chemistry | none   | 5      | acc    | 1.0000 | ± 0.0000 |
| - ceval-valid_middle_school_geography | none   | 5      | acc    | 0.8333 | ± 0.1124 |
| - ceval-valid_middle_school_history   | none   | 5      | acc    | 0.9545 | ± 0.0455 |
| - ceval-valid_middle_school_mathematics | none   | 5      | acc    | 0.8947 | ± 0.0723 |
| - ceval-valid_middle_school_physics   | none   | 5      | acc    | 0.9474 | ± 0.0526 |
| - ceval-valid_middle_school_politics  | none   | 5      | acc    | 0.9524 | ± 0.0476 |
| - ceval-valid_modern_chinese_history  | none   | 5      | acc    | 0.8696 | ± 0.0718 |
| - ceval-valid_operating_system        | none   | 5      | acc    | 0.8947 | ± 0.0723 |
| - ceval-valid_physician               | none   | 5      | acc    | 0.8163 | ± 0.0559 |
| - ceval-valid_plant_protection        | none   | 5      | acc    | 0.8182 | ± 0.0842 |
| - ceval-valid_probability_and_statistics | none   | 5      | acc    | 0.6111 | ± 0.1182 |
| - ceval-valid_professional_tour_guide | none   | 5      | acc    | 0.8621 | ± 0.0652 |
| - ceval-valid_sports_science          | none   | 5      | acc    | 1.0000 | ± 0.0000 |
| - ceval-valid_tax_accountant          | none   | 5      | acc    | 0.7551 | ± 0.0621 |
| - ceval-valid_teacher_qualification   | none   | 5      | acc    | 0.9545 | ± 0.0318 |
| - ceval-valid_urban_and_rural_planner | none   | 5      | acc    | 0.7609 | ± 0.0636 |
| - ceval-valid_veterinary_medicine     | none   | 5      | acc    | 0.9130 | ± 0.0601 |
</details>
<details>
<summary>gsm8k details</summary>

| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| gsm8k                                 | flexible-extract | 5      | exact_match | ✅0.8302 | ± 0.0103 |
</details>
