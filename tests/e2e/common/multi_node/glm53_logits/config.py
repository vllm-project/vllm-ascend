# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
import copy

CASES = ("disable_megamoe", "disable_pp", "disable_flashcomm1", "disable_lic8", "disable_sfac8")

COMMON = dict(
    trust_remote_code=True,
    dtype="bfloat16",
    quantization="ascend",
    distributed_executor_backend="mp",
    seed=1024,
    enable_expert_parallel=True,
    max_model_len=17408,
    max_num_seqs=4,
    gpu_memory_utilization=0.65,
    enable_prefix_caching=False,
    enable_chunked_prefill=True,
    async_scheduling=True,
    logprobs_mode="raw_logits",
    max_logprobs=-1,
    logits_processors=["tests.e2e.common.multi_node.glm53_logits.trace_processor:FixedTraceProcessor"],
    worker_extension_cls="tests.e2e.common.multi_node.glm53_logits.probe_extension.LogitsProbeExtension",
    kv_cache_dtype="fp8",
    attention_config={"indexer_kv_dtype": "fp8"},
)


def build(case, role, model):
    if case not in CASES or role not in ("P", "D"):
        raise ValueError("Unknown logits case or role")
    c = copy.deepcopy(COMMON)
    c.update(model=str(model), tokenizer=str(model))
    prefill = role == "P"
    c.update(
        tensor_parallel_size=2,
        pipeline_parallel_size=2 if prefill and case != "disable_pp" else 1,
        prefill_context_parallel_size=2 if prefill else 1,
        data_parallel_size=1 if prefill else 4,
        enforce_eager=prefill,
        max_num_batched_tokens=4096 if prefill else 256,
        speculative_config={
            "method": "deepseek_mtp",
            "num_speculative_tokens": 1 if prefill else 5,
            "enforce_eager": True,
        },
    )
    c["additional_config"] = {
        "enable_fused_mc2": 0 if case == "disable_megamoe" else 2,
        "enable_flashcomm1": prefill and case != "disable_flashcomm1",
        "enable_shared_expert_dp": True,
    }
    if not prefill:
        c["compilation_config"] = {"cudagraph_mode": "FULL_DECODE_ONLY"}
    if case == "disable_lic8":
        c["attention_config"]["indexer_kv_dtype"] = "bf16"
    if case == "disable_sfac8":
        c["kv_cache_dtype"] = "auto"
    c["kv_transfer_config"] = {
        "kv_connector": "MooncakeConnectorV2",
        "kv_role": "kv_producer" if prefill else "kv_consumer",
        "engine_id": "glm53-logits-p" if prefill else "glm53-logits-d",
        "kv_port": 29700 if prefill else 29800,
        "kv_connector_extra_config": {
            "use_ascend_direct": True,
            "prefill": {
                "dp_size": 1,
                "tp_size": 2,
                "pcp_size": 2,
                "pp_size": 1 if case == "disable_pp" else 2,
                **({"pp_layer_partition": "4,4"} if case != "disable_pp" else {}),
            },
            "decode": {"dp_size": 4, "tp_size": 2},
        },
    }
    return c
