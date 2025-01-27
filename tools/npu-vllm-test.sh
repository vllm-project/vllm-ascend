#!/bin/bash

# 
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -o pipefail

TEST_DIR="/home/cmq/code/vllm-cpu/vllm/tests"
TEST_FILES=(
    test_sequence.py
    # test_utils.py
    # test_config.py
    test_cache_block_hashing.py
    # test_scalartype.py
    # test_embedded_commit.py
    # test_inputs.py
    # test_sharded_state_loader.py
    test_sampling_params.py
    # test_logger.py
    # test_logits_processor.py
    # test_regression.py
    # prefix_caching/test_prefix_caching.py
    # prefix_caching/test_disable_sliding_window.py
    # weight_loading/test_weight_loading.py
    # samplers/test_beam_search.py
    # samplers/test_typical_acceptance_sampler.py
    # samplers/test_no_bad_words.py
    # samplers/test_rejection_sampler.py
    # samplers/test_ignore_eos.py
    # samplers/test_ranks.py
    # samplers/test_logits_processor.py
    # samplers/test_sampler.py
    # samplers/test_seeded_generate.py
    # samplers/test_logprobs.py
    # kernels/test_encoder_decoder_attn.py
    # kernels/test_rotary_embedding.py
    # kernels/test_prefix_prefill.py
    # kernels/test_flashinfer.py
    # kernels/utils.py
    # kernels/test_machete_mm.py
    # kernels/test_flash_attn.py
    # kernels/test_awq.py
    # kernels/test_blocksparse_attention.py
    # kernels/test_utils.py
    # kernels/test_aqlm.py
    # kernels/test_cutlass.py
    # kernels/test_causal_conv1d.py
    # kernels/test_marlin_gemm.py
    # kernels/test_layernorm.py
    # kernels/test_pos_encoding.py
    # kernels/test_moe.py
    # kernels/test_awq_marlin.py
    # kernels/test_int8_quant.py
    # kernels/test_gptq.py
    # kernels/test_attention.py
    # kernels/test_activation.py
    # kernels/quant_utils.py
    # kernels/test_permute_cols.py
    # kernels/test_triton_scaled_mm.py
    # kernels/test_gguf.py
    # kernels/test_awq_triton.py
    # kernels/test_attention_selector.py
    # kernels/test_ggml.py
    # kernels/test_mamba_ssm.py
    # kernels/test_fused_quant_layernorm.py
    # kernels/test_fp8_quant.py
    # kernels/test_cascade_flash_attn.py
    # kernels/conftest.py
    # kernels/allclose_default.py
    # kernels/test_block_fp8.py
    # kernels/test_cache.py
    # kernels/test_semi_structured.py
    # quantization/test_quark.py
    # quantization/test_compressed_tensors.py
    # quantization/utils.py
    # quantization/test_experts_int8.py
    # quantization/test_lm_head.py
    # quantization/test_ipex_quant.py
    # quantization/test_bitsandbytes.py
    # quantization/test_cpu_offload.py
    # quantization/test_fp8.py
    # quantization/test_configs.py
    # tool_use/test_tool_calls.py
    # tool_use/utils.py
    # tool_use/test_chat_completions.py
    # tool_use/test_jamba_tool_parser.py
    # tool_use/test_chat_completion_request_validations.py
    # tool_use/conftest.py
    # tool_use/test_parallel_tool_calls.py
    # runai_model_streamer/test_runai_model_streamer_loader.py
    # runai_model_streamer/test_weight_utils.py
    # kv_transfer/test_lookup_buffer.sh
    # kv_transfer/test_send_recv.py
    # kv_transfer/test_send_recv.sh
    # kv_transfer/test_lookup_buffer.py
    # kv_transfer/module_test.py
    # kv_transfer/disagg_test.py
    # plugins/vllm_add_dummy_platform/setup.py
    # plugins/vllm_add_dummy_platform/vllm_add_dummy_platform/dummy_platform.py
    # plugins/vllm_add_dummy_platform/vllm_add_dummy_platform/dummy_attention_backend.py
    # plugins/vllm_add_dummy_model/setup.py
    # plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_opt.py
    # plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_gemma_embedding.py
    # plugins/vllm_add_dummy_model/vllm_add_dummy_model/my_llava.py
    # prompt_adapter/test_multi_adapter_inference.py
    # prompt_adapter/test_pa_lora.py
    # prompt_adapter/test_bloom.py
    # compile/test_pass_manager.py
    # compile/utils.py
    # compile/test_wrapper.py
    # compile/test_fusion.py
    # compile/backend.py
    # compile/test_full_graph.py
    # compile/test_basic_correctness.py
    # compile/test_functionalization.py
    # compile/piecewise/test_simple.py
    # compile/piecewise/test_toy_llama.py
    # lora/test_punica_ops_variation.py
    # lora/test_quant_model.py
    # lora/test_lora_checkpoints.py
    # lora/test_mixtral.py
    # lora/test_qwen2vl.py
    # lora/test_baichuan.py
    # lora/utils.py
    # lora/test_phi.py
    # lora/test_utils.py
    # lora/test_minicpmv_tp.py
    # lora/test_layers.py
    # lora/test_worker.py
    # lora/test_jamba.py
    # lora/test_tokenizer_group.py
    # lora/test_lora_bias_e2e.py
    # lora/test_chatglm3_tp.py
    # lora/test_punica_ops_sizes.py
    # lora/test_lora_manager.py
    # lora/test_llama_tp.py
    # lora/test_lora_huggingface.py
    # lora/test_long_context.py
    # lora/test_gemma.py
    # lora/conftest.py
    # lora/data/long_context_test_data.py
    # models/registry.py
    # models/utils.py
    # models/test_registry.py
    # models/test_initialization.py
    # models/test_oot_registration.py
    # models/multimodal/processing/test_internvl.py
    # models/multimodal/processing/test_llava_next.py
    # models/multimodal/processing/test_idefics3.py
    # models/multimodal/processing/test_qwen2_vl.py
    # models/multimodal/processing/test_phi3v.py
    # models/multimodal/processing/test_common.py
    # models/multimodal/processing/test_qwen.py
    # models/multimodal/processing/test_llava_onevision.py
    # models/encoder_decoder/language/test_bart.py
    # models/encoder_decoder/audio_language/test_whisper.py
    # models/encoder_decoder/vision_language/test_broadcast.py
    # models/encoder_decoder/vision_language/test_florence2.py
    # models/encoder_decoder/vision_language/test_mllama.py
    # models/decoder_only/language/test_models.py
    # models/decoder_only/language/test_gptq_marlin.py
    # models/decoder_only/language/test_granite.py
    # models/decoder_only/language/test_modelopt.py
    # models/decoder_only/language/test_phimoe.py
    # models/decoder_only/language/test_aqlm.py
    # models/decoder_only/language/test_mistral.py
    # models/decoder_only/language/test_jamba.py
    # models/decoder_only/language/test_gptq_marlin_24.py
    # models/decoder_only/language/test_mamba.py
    # models/decoder_only/language/test_gguf.py
    # models/decoder_only/language/test_fp8.py
    # models/decoder_only/audio_language/test_ultravox.py
    # models/decoder_only/vision_language/test_models.py
    # models/decoder_only/vision_language/test_awq.py
    # models/decoder_only/vision_language/test_intern_vit.py
    # models/decoder_only/vision_language/test_qwen2_vl.py
    # models/decoder_only/vision_language/test_pixtral.py
    # models/decoder_only/vision_language/test_phi3v.py
    # models/decoder_only/vision_language/test_h2ovl.py
    # models/decoder_only/vision_language/vlm_utils/types.py
    # models/decoder_only/vision_language/vlm_utils/model_utils.py
    # models/decoder_only/vision_language/vlm_utils/runners.py
    # models/decoder_only/vision_language/vlm_utils/core.py
    # models/decoder_only/vision_language/vlm_utils/custom_inputs.py
    # models/decoder_only/vision_language/vlm_utils/case_filtering.py
    # models/decoder_only/vision_language/vlm_utils/builders.py
    # models/embedding/utils.py
    # models/embedding/language/test_scoring.py
    # models/embedding/language/test_gritlm.py
    # models/embedding/language/test_cls_models.py
    # models/embedding/language/test_embedding.py
    # models/embedding/vision_language/test_llava_next.py
    # models/embedding/vision_language/test_dse_qwen2_vl.py
    # models/embedding/vision_language/test_phi3v.py
    # multimodal/utils.py
    # multimodal/test_processor_kwargs.py
    # multimodal/test_utils.py
    # multimodal/test_inputs.py
    # multimodal/test_processing.py
    # standalone_tests/python_only_compile.sh
    # standalone_tests/lazy_torch_compile.py
    # async_engine/test_async_llm_engine.py
    # async_engine/api_server_async_engine.py
    # async_engine/test_api_server.py
    # async_engine/test_request_tracker.py
    # mq_llm_engine/utils.py
    # mq_llm_engine/test_load.py
    # mq_llm_engine/test_abort.py
    # mq_llm_engine/test_error_handling.py
    # tokenization/test_tokenizer.py
    # tokenization/test_tokenizer_group.py
    # tokenization/test_get_eos.py
    # tokenization/test_cached_tokenizer.py
    # tokenization/test_detokenize.py
    # core/utils.py
    # core/test_chunked_prefill_scheduler.py
    # core/test_serialization.py
    # core/test_num_computed_tokens_update.py
    # core/test_scheduler_encoder_decoder.py
    # core/test_scheduler.py
    # core/block/test_cpu_gpu_block_allocator.py
    # core/block/test_prefix_caching_block.py
    # core/block/test_common.py
    # core/block/test_block_table.py
    # core/block/test_block_manager.py
    # core/block/conftest.py
    # core/block/test_naive_block.py
    # core/block/e2e/test_correctness.py
    # core/block/e2e/test_correctness_sliding_window.py
    # core/block/e2e/conftest.py
    # tracing/test_tracing.py
    # engine/test_arg_utils.py
    # engine/test_detokenization.py
    # engine/test_short_mm_context.py
    # engine/test_custom_executor.py
    # engine/test_multiproc_workers.py
    # engine/test_computed_prefix_blocks.py
    # engine/test_stop_reason.py
    # engine/test_skip_tokenizer_init.py
    # engine/test_stop_strings.py
    # engine/output_processor/test_stop_checker.py
    # engine/output_processor/test_multi_step.py
    # tensorizer_loader/test_tensorizer.py
    # tensorizer_loader/conftest.py
    # entrypoints/test_chat_utils.py
    # entrypoints/conftest.py
    # entrypoints/llm/test_lazy_outlines.py
    # entrypoints/llm/test_generate_multiple_loras.py
    # entrypoints/llm/test_encode.py
    # entrypoints/llm/test_init.py
    # entrypoints/llm/test_guided_generate.py
    # entrypoints/llm/test_gpu_utilization.py
    # entrypoints/llm/test_chat.py
    # entrypoints/llm/test_accuracy.py
    # entrypoints/llm/test_prompt_validation.py
    # entrypoints/llm/test_generate.py
    # entrypoints/offline_mode/test_offline_mode.py
    # entrypoints/openai/test_completion.py
    # entrypoints/openai/test_models.py
    # entrypoints/openai/test_chat_echo.py
    # entrypoints/openai/test_score.py
    # entrypoints/openai/test_tokenization.py
    # entrypoints/openai/test_cli_args.py
    # entrypoints/openai/test_chunked_prompt.py
    # entrypoints/openai/test_encoder_decoder.py
    # entrypoints/openai/test_chat_template.py
    # entrypoints/openai/test_oot_registration.py
    # entrypoints/openai/test_run_batch.py
    # entrypoints/openai/test_metrics.py
    # entrypoints/openai/test_vision_embedding.py
    # entrypoints/openai/test_embedding.py
    # entrypoints/openai/test_lora_adapters.py
    # entrypoints/openai/test_video.py
    # entrypoints/openai/test_serving_models.py
    # entrypoints/openai/test_chat.py
    # entrypoints/openai/test_pooling.py
    # entrypoints/openai/test_basic.py
    # entrypoints/openai/test_accuracy.py
    # entrypoints/openai/test_prompt_validation.py
    # entrypoints/openai/test_vision.py
    # entrypoints/openai/test_audio.py
    # entrypoints/openai/test_async_tokenization.py
    # entrypoints/openai/test_return_tokens_as_ids.py
    # entrypoints/openai/test_serving_chat.py
    # entrypoints/openai/test_shutdown.py
    # entrypoints/openai/test_root_path.py
    # entrypoints/openai/tool_parsers/utils.py
    # entrypoints/openai/tool_parsers/test_pythonic_tool_parser.py
    # model_executor/weight_utils.py
    # model_executor/test_enabled_custom_ops.py
    # model_executor/test_guided_processors.py
    # model_executor/test_model_load_with_params.py
    # model_executor/conftest.py
    # metrics/test_metrics.py
    # system_messages/sonnet3.5_nov2024.txt
    # encoder_decoder/test_e2e_correctness.py
    # v1/core/test_kv_cache_utils.py
    # v1/core/test_prefix_caching.py
    # v1/sample/test_sampler.py
    # v1/engine/test_engine_core.py
    # v1/engine/test_async_llm.py
    # v1/engine/test_output_processor.py
    # v1/engine/test_engine_args.py
    # v1/engine/test_engine_core_client.py
    # v1/e2e/test_cascade_attention.py
    # v1/worker/test_gpu_input_batch.py
    # spec_decode/utils.py
    # spec_decode/test_utils.py
    # spec_decode/test_ngram_worker.py
    # spec_decode/test_metrics.py
    # spec_decode/test_batch_expansion.py
    # spec_decode/test_multi_step_worker.py
    # spec_decode/test_scorer.py
    # spec_decode/test_spec_decode_worker.py
    # spec_decode/test_dynamic_spec_decode.py
    # spec_decode/e2e/test_mlp_correctness.py
    # spec_decode/e2e/test_ngram_correctness.py
    # spec_decode/e2e/test_seed.py
    # spec_decode/e2e/test_integration.py
    # spec_decode/e2e/test_medusa_correctness.py
    # spec_decode/e2e/test_integration_dist_tp4.py
    # spec_decode/e2e/test_eagle_correctness.py
    # spec_decode/e2e/test_compatibility.py
    # spec_decode/e2e/test_multistep_correctness.py
    # spec_decode/e2e/test_integration_dist_tp2.py
    # spec_decode/e2e/conftest.py
    # spec_decode/e2e/test_logprobs.py
    # multi_step/test_correctness_async_llm.py
    # multi_step/test_correctness_llm.py
    # vllm_test_utils/setup.py
    # vllm_test_utils/vllm_test_utils/blame.py
    # vllm_test_utils/vllm_test_utils/monitor.py
    # plugins_tests/test_platform_plugins.py
    # tpu/test_compilation.py
    # tpu/test_quantization_accuracy.py
    # tpu/test_custom_dispatcher.py
    # distributed/test_custom_all_reduce.py
    # distributed/test_distributed_oot.py
    # distributed/test_pipeline_parallel.py
    # distributed/test_pynccl.py
    # distributed/test_pipeline_partition.py
    # distributed/test_utils.py
    # distributed/test_pp_cudagraph.py
    # distributed/test_ca_buffer_sharing.py
    # distributed/test_multi_node_assignment.py
    # distributed/test_same_node.py
    # distributed/test_shm_broadcast.py
    # distributed/test_comm_ops.py
    # basic_correctness/test_chunked_prefill.py
    # basic_correctness/test_preemption.py
    # basic_correctness/test_cpu_offload.py
    # basic_correctness/test_basic_correctness.py
    # worker/test_model_runner.py
    # worker/test_encoder_decoder_model_runner.py
    # worker/test_swap.py
    # worker/test_profile.py
    # worker/test_model_input.py
)

# print usage
usage() {
    echo "Usage: $0 -t <test_script_1> -t <test_script_2> ..."
    echo "Example: $0 -t test_inputs.py -t test_regression.py"
    exit 1
}

# parse command line args
while getopts ":t:" opt; do
    case ${opt} in
        t)
            TEST_FILES+=("${OPTARG}")
            ;;
        *)
            usage
            ;;
    esac
done

echo "------ Test vllm_ascend on vLLM native ut ------"


# check if the test scripts are specified
if [ ${#TEST_FILES[@]} -eq 0 ]; then
    echo "Error: No test scripts specified."
    usage
fi


# test all the specified ut
for test_file in "${TEST_FILES[@]}"; do
    full_path="$TEST_DIR/$test_file"
    if [ -f "$full_path" ]; then
        echo "Running $test_file..."
        pytest -sv "$full_path"

        # Check if pytest ran successfully
        if [ $? -ne 0 ]; then
            echo "Error: $test_file failed."
            exit 1
        fi

        echo "Completed $test_file."
    else
        echo "Error: $test_file not found in $TEST_DIR."
        exit 1
    fi
done

echo "------ All specified tests completed -------"
