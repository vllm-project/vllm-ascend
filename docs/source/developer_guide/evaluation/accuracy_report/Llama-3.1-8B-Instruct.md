# Llama-3.1-8B-Instruct
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
      <strong>Datasets</strong>: ceval-valid,mmlu,gsm8k <br>
  </div>
  <div>
      <strong>Command</strong>:

  ```bash
  export MODEL_AEGS='meta-llama/Llama-3.1-8B-Instruct, max_model_len=4096,dtype=auto,tensor_parallel_size=2,gpu_memory_utilization=0.6'
lm_eval --model vllm --modlel_args $MODEL_ARGS --tasks ceval-valid,gsm8k \
--apply_chat_template --fewshot_as_multiturn --num_fewshot 5 --batch_size 1
  ```
  </div>
  <div>&nbsp;</div>

| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| ceval-valid                           | none   | 5      | acc    | ↑ 0.5483 | ± 0.0132 |
| mmlu                                  | none   | 5      | acc    | ↑ 0.6867 | ± 0.0037 |
| gsm8k                                 | flexible-extract | 5      | exact_match | ↑ 0.7286 | ± 0.0122 |
<details>
<summary>ceval-valid</summary>

| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| ceval-valid                           | none   | 5      | acc    | ↑ 0.5483 | ± 0.0132 |
| - ceval-valid_accountant              | none   | 5      | acc    | ↑ 0.4898 | ± 0.0722 |
| - ceval-valid_advanced_mathematics    | none   | 5      | acc    | ↑ 0.5263 | ± 0.1177 |
| - ceval-valid_art_studies             | none   | 5      | acc    | ↑ 0.5455 | ± 0.0880 |
| - ceval-valid_basic_medicine          | none   | 5      | acc    | ↑ 0.6842 | ± 0.1096 |
| - ceval-valid_business_administration | none   | 5      | acc    | ↑ 0.6061 | ± 0.0864 |
| - ceval-valid_chinese_language_and_literature | none   | 5      | acc    | ↑ 0.4348 | ± 0.1057 |
| - ceval-valid_civil_servant           | none   | 5      | acc    | ↑ 0.4894 | ± 0.0737 |
| - ceval-valid_clinical_medicine       | none   | 5      | acc    | ↑ 0.5455 | ± 0.1087 |
| - ceval-valid_college_chemistry       | none   | 5      | acc    | ↑ 0.4167 | ± 0.1028 |
| - ceval-valid_college_economics       | none   | 5      | acc    | ↑ 0.4545 | ± 0.0678 |
| - ceval-valid_college_physics         | none   | 5      | acc    | ↑ 0.4737 | ± 0.1177 |
| - ceval-valid_college_programming     | none   | 5      | acc    | ↑ 0.5946 | ± 0.0818 |
| - ceval-valid_computer_architecture   | none   | 5      | acc    | ↑ 0.5714 | ± 0.1107 |
| - ceval-valid_computer_network        | none   | 5      | acc    | ↑ 0.7895 | ± 0.0961 |
| - ceval-valid_discrete_mathematics    | none   | 5      | acc    | ↑ 0.4375 | ± 0.1281 |
| - ceval-valid_education_science       | none   | 5      | acc    | ↑ 0.7241 | ± 0.0845 |
| - ceval-valid_electrical_engineer     | none   | 5      | acc    | ↑ 0.4324 | ± 0.0826 |
| - ceval-valid_environmental_impact_assessment_engineer | none   | 5      | acc    | ↑ 0.5484 | ± 0.0909 |
| - ceval-valid_fire_engineer           | none   | 5      | acc    | ↑ 0.4839 | ± 0.0912 |
| - ceval-valid_high_school_biology     | none   | 5      | acc    | ↑ 0.5263 | ± 0.1177 |
| - ceval-valid_high_school_chemistry   | none   | 5      | acc    | ↑ 0.4737 | ± 0.1177 |
| - ceval-valid_high_school_chinese     | none   | 5      | acc    | ↑ 0.2105 | ± 0.0961 |
| - ceval-valid_high_school_geography   | none   | 5      | acc    | ↑ 0.6842 | ± 0.1096 |
| - ceval-valid_high_school_history     | none   | 5      | acc    | ↑ 0.6500 | ± 0.1094 |
| - ceval-valid_high_school_mathematics | none   | 5      | acc    | ↑ 0.0000 | ± 0.0000 |
| - ceval-valid_high_school_physics     | none   | 5      | acc    | ↑ 0.3158 | ± 0.1096 |
| - ceval-valid_high_school_politics    | none   | 5      | acc    | ↑ 0.5789 | ± 0.1164 |
| - ceval-valid_ideological_and_moral_cultivation | none   | 5      | acc    | ↑ 0.8947 | ± 0.0723 |
| - ceval-valid_law                     | none   | 5      | acc    | ↑ 0.4583 | ± 0.1039 |
| - ceval-valid_legal_professional      | none   | 5      | acc    | ↑ 0.3913 | ± 0.1041 |
| - ceval-valid_logic                   | none   | 5      | acc    | ↑ 0.5000 | ± 0.1091 |
| - ceval-valid_mao_zedong_thought      | none   | 5      | acc    | ↑ 0.5000 | ± 0.1043 |
| - ceval-valid_marxism                 | none   | 5      | acc    | ↑ 0.6842 | ± 0.1096 |
| - ceval-valid_metrology_engineer      | none   | 5      | acc    | ↑ 0.5833 | ± 0.1028 |
| - ceval-valid_middle_school_biology   | none   | 5      | acc    | ↑ 0.7143 | ± 0.1010 |
| - ceval-valid_middle_school_chemistry | none   | 5      | acc    | ↑ 0.8500 | ± 0.0819 |
| - ceval-valid_middle_school_geography | none   | 5      | acc    | ↑ 0.5833 | ± 0.1486 |
| - ceval-valid_middle_school_history   | none   | 5      | acc    | ↑ 0.5455 | ± 0.1087 |
| - ceval-valid_middle_school_mathematics | none   | 5      | acc    | ↑ 0.3684 | ± 0.1137 |
| - ceval-valid_middle_school_physics   | none   | 5      | acc    | ↑ 0.6316 | ± 0.1137 |
| - ceval-valid_middle_school_politics  | none   | 5      | acc    | ↑ 0.8095 | ± 0.0878 |
| - ceval-valid_modern_chinese_history  | none   | 5      | acc    | ↑ 0.5217 | ± 0.1065 |
| - ceval-valid_operating_system        | none   | 5      | acc    | ↑ 0.6316 | ± 0.1137 |
| - ceval-valid_physician               | none   | 5      | acc    | ↑ 0.5918 | ± 0.0709 |
| - ceval-valid_plant_protection        | none   | 5      | acc    | ↑ 0.7727 | ± 0.0914 |
| - ceval-valid_probability_and_statistics | none   | 5      | acc    | ↑ 0.3889 | ± 0.1182 |
| - ceval-valid_professional_tour_guide | none   | 5      | acc    | ↑ 0.6207 | ± 0.0917 |
| - ceval-valid_sports_science          | none   | 5      | acc    | ↑ 0.6316 | ± 0.1137 |
| - ceval-valid_tax_accountant          | none   | 5      | acc    | ↑ 0.3878 | ± 0.0703 |
| - ceval-valid_teacher_qualification   | none   | 5      | acc    | ↑ 0.7955 | ± 0.0615 |
| - ceval-valid_urban_and_rural_planner | none   | 5      | acc    | ↑ 0.5217 | ± 0.0745 |
| - ceval-valid_veterinary_medicine     | none   | 5      | acc    | ↑ 0.6087 | ± 0.1041 |
</details>
<details>
<summary>mmlu</summary>

| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| mmlu                                  | none   | 5      | acc    | ↑ 0.6867 | ± 0.0037 |
| - humanities                          | none   | 5      | acc    | ↑ 0.6495 | ± 0.0067 |
| - formal_logic                        | none   | 5      | acc    | ↑ 0.5714 | ± 0.0443 |
| - high_school_european_history        | none   | 5      | acc    | ↑ 0.7636 | ± 0.0332 |
| - high_school_us_history              | none   | 5      | acc    | ↑ 0.8186 | ± 0.0270 |
| - high_school_world_history           | none   | 5      | acc    | ↑ 0.8439 | ± 0.0236 |
| - international_law                   | none   | 5      | acc    | ↑ 0.8347 | ± 0.0339 |
| - jurisprudence                       | none   | 5      | acc    | ↑ 0.7778 | ± 0.0402 |
| - logical_fallacies                   | none   | 5      | acc    | ↑ 0.8098 | ± 0.0308 |
| - moral_disputes                      | none   | 5      | acc    | ↑ 0.7630 | ± 0.0229 |
| - moral_scenarios                     | none   | 5      | acc    | ↑ 0.5687 | ± 0.0166 |
| - philosophy                          | none   | 5      | acc    | ↑ 0.7363 | ± 0.0250 |
| - prehistory                          | none   | 5      | acc    | ↑ 0.7562 | ± 0.0239 |
| - professional_law                    | none   | 5      | acc    | ↑ 0.5111 | ± 0.0128 |
| - world_religions                     | none   | 5      | acc    | ↑ 0.8363 | ± 0.0284 |
| - other                               | none   | 5      | acc    | ↑ 0.7448 | ± 0.0075 |
| - business_ethics                     | none   | 5      | acc    | ↑ 0.7200 | ± 0.0451 |
| - clinical_knowledge                  | none   | 5      | acc    | ↑ 0.7509 | ± 0.0266 |
| - college_medicine                    | none   | 5      | acc    | ↑ 0.6821 | ± 0.0355 |
| - global_facts                        | none   | 5      | acc    | ↑ 0.3900 | ± 0.0490 |
| - human_aging                         | none   | 5      | acc    | ↑ 0.6951 | ± 0.0309 |
| - management                          | none   | 5      | acc    | ↑ 0.8155 | ± 0.0384 |
| - marketing                           | none   | 5      | acc    | ↑ 0.8974 | ± 0.0199 |
| - medical_genetics                    | none   | 5      | acc    | ↑ 0.8200 | ± 0.0386 |
| - miscellaneous                       | none   | 5      | acc    | ↑ 0.8378 | ± 0.0132 |
| - nutrition                           | none   | 5      | acc    | ↑ 0.8039 | ± 0.0227 |
| - professional_accounting             | none   | 5      | acc    | ↑ 0.5532 | ± 0.0297 |
| - professional_medicine               | none   | 5      | acc    | ↑ 0.7721 | ± 0.0255 |
| - virology                            | none   | 5      | acc    | ↑ 0.5241 | ± 0.0389 |
| - social sciences                     | none   | 5      | acc    | ↑ 0.7797 | ± 0.0073 |
| - econometrics                        | none   | 5      | acc    | ↑ 0.6053 | ± 0.0460 |
| - high_school_geography               | none   | 5      | acc    | ↑ 0.8485 | ± 0.0255 |
| - high_school_government_and_politics | none   | 5      | acc    | ↑ 0.9171 | ± 0.0199 |
| - high_school_macroeconomics          | none   | 5      | acc    | ↑ 0.6923 | ± 0.0234 |
| - high_school_microeconomics          | none   | 5      | acc    | ↑ 0.7647 | ± 0.0276 |
| - high_school_psychology              | none   | 5      | acc    | ↑ 0.8697 | ± 0.0144 |
| - human_sexuality                     | none   | 5      | acc    | ↑ 0.8015 | ± 0.0350 |
| - professional_psychology             | none   | 5      | acc    | ↑ 0.7271 | ± 0.0180 |
| - public_relations                    | none   | 5      | acc    | ↑ 0.6818 | ± 0.0446 |
| - security_studies                    | none   | 5      | acc    | ↑ 0.7224 | ± 0.0287 |
| - sociology                           | none   | 5      | acc    | ↑ 0.8358 | ± 0.0262 |
| - us_foreign_policy                   | none   | 5      | acc    | ↑ 0.8900 | ± 0.0314 |
| - stem                                | none   | 5      | acc    | ↑ 0.5940 | ± 0.0084 |
| - abstract_algebra                    | none   | 5      | acc    | ↑ 0.3900 | ± 0.0490 |
| - anatomy                             | none   | 5      | acc    | ↑ 0.6741 | ± 0.0405 |
| - astronomy                           | none   | 5      | acc    | ↑ 0.7566 | ± 0.0349 |
| - college_biology                     | none   | 5      | acc    | ↑ 0.8264 | ± 0.0317 |
| - college_chemistry                   | none   | 5      | acc    | ↑ 0.4700 | ± 0.0502 |
| - college_computer_science            | none   | 5      | acc    | ↑ 0.5400 | ± 0.0501 |
| - college_mathematics                 | none   | 5      | acc    | ↑ 0.3900 | ± 0.0490 |
| - college_physics                     | none   | 5      | acc    | ↑ 0.4314 | ± 0.0493 |
| - computer_security                   | none   | 5      | acc    | ↑ 0.8000 | ± 0.0402 |
| - conceptual_physics                  | none   | 5      | acc    | ↑ 0.6170 | ± 0.0318 |
| - electrical_engineering              | none   | 5      | acc    | ↑ 0.6552 | ± 0.0396 |
| - elementary_mathematics              | none   | 5      | acc    | ↑ 0.4735 | ± 0.0257 |
| - high_school_biology                 | none   | 5      | acc    | ↑ 0.8097 | ± 0.0223 |
| - high_school_chemistry               | none   | 5      | acc    | ↑ 0.6207 | ± 0.0341 |
| - high_school_computer_science        | none   | 5      | acc    | ↑ 0.7300 | ± 0.0446 |
| - high_school_mathematics             | none   | 5      | acc    | ↑ 0.4222 | ± 0.0301 |
| - high_school_physics                 | none   | 5      | acc    | ↑ 0.4636 | ± 0.0407 |
| - high_school_statistics              | none   | 5      | acc    | ↑ 0.6065 | ± 0.0333 |
| - machine_learning                    | none   | 5      | acc    | ↑ 0.5446 | ± 0.0473 |
</details>
<details>
<summary>gsm8k</summary>

| Task                  | Filter | n-shot | Metric   | Value   | Stderr |
|-----------------------|-------:|-------:|----------|--------:|-------:|
| gsm8k                                 | flexible-extract | 5      | exact_match | ↑ 0.7286 | ± 0.0122 |
</details>