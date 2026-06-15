#!/usr/bin/env python3
"""Generate coverage.md by scanning test files (Python + YAML) and
extracting feature flags via AST and YAML parsing."""

import ast
import contextlib
import json
from pathlib import Path
from typing import Any

import regex as re

REPO_ROOT = Path(__file__).resolve().parents[2]
E2E_PR_ROOT = REPO_ROOT / "tests" / "e2e" / "pull_request"
OUTPUT_FILE = Path(__file__).resolve().parent / "coverage.md"

COLUMNS = [
    "Test file",
    "Test method",
    "Model",
    "310P",
    "Dense",
    "MoE",
    "Embedding",
    "Classification",
    "Reranker",
    "Mamba/SSM",
    "Multimodal Reasoning",
    "TP",
    "PP",
    "EP",
    "PCP",
    "DCP",
    "Context Parallel",
    "EPLB",
    "Dynamic EPLB",
    "Multistream MoE",
    "Full Graph",
    "Full Decode Only Graph",
    "Default FULL_AND_PIECEWISE Graph",
    "Piecewise Graph",
    "Eager Mode",
    "PD disaggregation",
    "W8A8",
    "W4A8",
    "FP16",
    "LoRA",
    "Multi-LoRA",
    "Runtime LoRA updating",
    "Fully sharded LoRA parameterization",
    "Spec Decode",
    "MTP",
    "Eagle-3",
    "SFA/DSA",
    "DSA CP",
    "Pooling runner",
    "Score API",
    "Classification API",
    "Distributed executor mp",
    "Flash Attention 3",
    "FIA comparison",
    "Chunked Prefill",
    "Prefix Caching",
    "CPU/KV offloading",
    "KV transfer/events",
    "Sleep/Wake memory",
    "Xlite Graph",
    "CP KV Interleave",
    "Long Sequence",
    "FlashComm1 env",
    "Skipped",
    "Conditional skip",
    "Logprobs",
    "Batch inference",
    "Mixed lengths",
]

CARD_SECTIONS = [
    ("1-Card Tests", "one_card"),
    ("2-Card Tests", "two_card"),
    ("4-Card Tests", "four_card"),
]

CHECK = "\u2705"
EMPTY = ""


def _source_to_str(node):
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.Str):
        return node.s
    return None


def _flatten_dict_keys(d):
    out = set()
    if isinstance(d, dict):
        for k, v in d.items():
            out.add(k)
            if isinstance(v, dict):
                out.update(_flatten_dict_keys(v))
    return out


def _extract_kwarg_dict(call_node, key):
    for kw in call_node.keywords:
        if kw.arg == key and isinstance(kw.value, ast.Dict):
            d = {}
            for dk, dv in zip(kw.value.keys, kw.value.values):
                ks = _source_to_str(dk)
                vs = _source_to_str(dv)
                if ks is not None:
                    d[ks] = vs
                elif isinstance(dk, ast.Constant):
                    ks2 = str(dk.value)
                    d[ks2] = vs
            return d
    return None


def _extract_kwarg_value(call_node, key):
    for kw in call_node.keywords:
        if kw.arg == key:
            return _source_to_str(kw.value)
    return None


def _extract_kwarg_int(call_node, key):
    for kw in call_node.keywords:
        if kw.arg == key and isinstance(kw.value, ast.Constant):
            try:
                return int(kw.value.value)
            except (TypeError, ValueError):
                return None
    return None


def _extract_kwarg_bool(call_node, key):
    for kw in call_node.keywords:
        if kw.arg == key and isinstance(kw.value, ast.Constant):
            return bool(kw.value.value)
    return None


def _find_calls_in_body(body, func_name):
    calls = []
    for node in _walk_body(body):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == func_name
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == func_name
            ):
                calls.append(node)
    return calls


def _find_all_calls_in_body(body):
    calls = []
    for node in _walk_body(body):
        if isinstance(node, ast.Call):
            calls.append(node)
    return calls


def _extract_model_from_call(call_node):
    model = _extract_kwarg_value(call_node, "model_name") or _extract_kwarg_value(call_node, "model")
    if model:
        return model
    pos_args = call_node.args
    if pos_args and isinstance(pos_args[0], ast.Constant):
        return str(pos_args[0].value)
    return None


def _extract_config_from_call(call_node):
    config_keys = set()
    for kw in call_node.keywords:
        config_keys.add(kw.arg)
    comp_config = _extract_kwarg_dict(call_node, "compilation_config")
    if comp_config:
        config_keys.update(_flatten_dict_keys(comp_config))
    spec_config = _extract_kwarg_dict(call_node, "speculative_config")
    if spec_config:
        config_keys.update(_flatten_dict_keys(spec_config))
    add_config = _extract_kwarg_dict(call_node, "additional_config")
    if add_config:
        config_keys.update(_flatten_dict_keys(add_config))
    return config_keys


def _detect_cards(rel_path):
    parts = rel_path.split("/")
    for p in parts:
        if p == "one_card":
            return 1
        if p == "two_card":
            return 2
        if p == "four_card":
            return 4
    return 1


def _detect_310p(rel_path):
    return "_310p" in rel_path or "/310p/" in rel_path or rel_path.startswith("310p/")


def _classify_model(model_name):
    flags = {
        "Dense": False,
        "MoE": False,
        "Embedding": False,
        "Classification": False,
        "Reranker": False,
        "Mamba/SSM": False,
        "Multimodal Reasoning": False,
    }
    if not model_name:
        return flags
    ml = model_name.lower()
    is_moe = any(
        x in model_name
        for x in [
            "A3B",
            "MoE",
            "DeepSeek-V2",
            "DeepSeek-V3",
            "DeepSeek-V4",
            "Qwen3-30B-A3B",
            "Qwen3.5-35B-A3B",
            "Qwen3-Next-80B-A3B",
            "Qwen3-Coder-30B-A3B",
        ]
    )
    is_embedding = any(x in ml for x in ["embedding", "e5", "bge-m3", "minilm"]) or any(
        x in model_name for x in ["Embedding"]
    )
    is_classification = any(x in ml for x in ["apeach", "classification"]) or "SequenceClassification" in model_name
    is_reranker = any(x in ml for x in ["reranker"])
    is_mamba_ssm = "Qwen3.5" in model_name and "VL" not in model_name
    is_vl = any(x in model_name for x in ["VL", "HunyuanOCR", "Audio", "whisper", "MiniCPM-V"])
    if is_vl:
        flags["Multimodal Reasoning"] = True
    if is_mamba_ssm and not is_vl:
        flags["Mamba/SSM"] = True
    if is_reranker:
        flags["Reranker"] = True
    if is_classification:
        flags["Classification"] = True
    if is_embedding:
        flags["Embedding"] = True
    if is_moe:
        flags["MoE"] = True
    is_other = not (is_moe or is_embedding or is_classification or is_reranker or is_vl)
    if is_other and not is_mamba_ssm:
        flags["Dense"] = True
    return flags


def _extract_env_vars(body):
    env_vars = {}
    for node in _walk_body(body):
        if isinstance(node, ast.Call):
            fn = None
            if isinstance(node.func, ast.Name):
                fn = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fn = node.func.attr
            if fn == "setenv" and node.args:
                key = _source_to_str(node.args[0])
                val = _source_to_str(node.args[1]) if len(node.args) > 1 else None
                if key:
                    env_vars[key] = val
            if fn == "patch" and isinstance(node.func, ast.Attribute):
                pass
    for node in _walk_body(body):
        if isinstance(node, ast.Subscript):
            try:
                if (
                    (isinstance(node.value, ast.Attribute) and node.value.attr == "environ")
                    or isinstance(node.value, ast.Name)
                    and node.value.id == "environ"
                ):
                    sl = node.slice
                    if isinstance(sl, ast.Constant):
                        env_vars[str(sl.value)] = None
            except Exception:
                pass
    for node in _walk_body(body):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "dict" and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "patch":
                    for kw in node.keywords:
                        if kw.arg == "os.environ" and isinstance(kw.value, ast.Dict):
                            for dk, dv in zip(kw.value.keys, kw.value.values):
                                ks = _source_to_str(dk)
                                vs = _source_to_str(dv)
                                if ks:
                                    env_vars[ks] = vs
    return env_vars


def _extract_decorators(func_node):
    has_skip = False
    has_skipif = False
    for dec in func_node.decorator_list:
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
            if dec.func.id == "pytest.mark.skip":
                has_skip = True
            if dec.func.id == "pytest.mark.skipif":
                has_skipif = True
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
            if dec.func.attr == "skip":
                has_skip = True
            if dec.func.attr == "skipif":
                has_skipif = True
        if isinstance(dec, ast.Name) and dec.id == "pytest.mark.skip":
            has_skip = True
    return has_skip, has_skipif


def _walk_body(body):
    if isinstance(body, list):
        for stmt in body:
            yield from ast.walk(stmt)
    else:
        yield from ast.walk(body)


def _find_llm_or_vllmrunner_calls(body):
    calls = []
    for node in _walk_body(body):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in ("LLM", "VllmRunner")
                or isinstance(node.func, ast.Name)
                and node.func.id == "compare_logprobs"
                or isinstance(node.func, ast.Name)
                and node.func.id == "check_outputs_equal"
            ):
                calls.append(node)
    return calls


def _extract_all_kwarg_keys(body):
    keys = set()
    for node in _walk_body(body):
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg:
                    keys.add(kw.arg)
    return keys


def _find_method_calls(body):
    method_names = set()
    for node in _walk_body(body):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method_names.add(node.func.attr)
    return method_names


def _find_lora_requests(body):
    count = 0
    for node in _walk_body(body):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "LoRARequest":
                count += 1
    return count


def _extract_models_from_parametrize(func_node, source_code):
    models = []
    for dec in func_node.decorator_list:
        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute):
            if dec.func.attr == "parametrize":
                for kw in dec.keywords:
                    if kw.arg in ("model", "model_name", "model_setup") and isinstance(kw.value, ast.List):
                        for elt in kw.value.elts:
                            if isinstance(elt, ast.Constant):
                                val = elt.value
                                if isinstance(val, str):
                                    models.append(val)
                                elif isinstance(val, tuple):
                                    models.append(val[1] if len(val) > 1 else str(val))
                    elif kw.arg == "method" and isinstance(kw.value, ast.Dict):
                        pass
    return models


def _extract_server_args_models(body, source_code):
    models = set()
    for node in _walk_body(body):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in (
                "RemoteOpenAIServer",
                "RemotePDServer",
                "RemoteEPDServer",
            ):
                for arg in node.args:
                    if isinstance(arg, ast.Constant):
                        models.add(str(arg.value))
                for kw in node.keywords:
                    if kw.arg == "model" and isinstance(kw.value, ast.Constant):
                        models.add(str(kw.value.value))
    if not models:
        docstring_val = None
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            if isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                docstring_val = body[0].value.value
        str_literals = set()
        for node in _walk_body(body):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value
                if v == docstring_val:
                    continue
                if len(v) > 200:
                    continue
                if any(
                    p in v
                    for p in [
                        "Qwen",
                        "DeepSeek",
                        "MiniCPM",
                        "ilama",
                        "LLM-Research",
                        "gdydems",
                        "BAAI",
                        "intfloat",
                        "sentence-transformers",
                        "Howeee",
                        "vllm-ascend",
                        "charent",
                        "RedHatAI",
                        "z-lab",
                        "amd",
                        "MNN",
                        "wemaster",
                        "dengcao",
                        "openai-mirror",
                        "openbmb",
                        "OpenBMB",
                        "Meta-Llama",
                        "amazon",
                    ]
                ):
                    if not v.startswith("--") and "/" in v and len(v) > 5:
                        str_literals.add(v)
        models = str_literals
    return models


def _process_test_file(filepath, source_code, root_path=None):
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return []

    if root_path is None:
        root_path = E2E_PR_ROOT
    rel_path = str(filepath.relative_to(root_path))
    is_310p = _detect_310p(rel_path)

    card_prefix = ""
    for prefix in ("one_card/", "two_card/", "four_card/"):
        if rel_path.startswith(prefix):
            card_prefix = prefix
            break
    display_path = rel_path[len(card_prefix) :] if card_prefix else rel_path

    results = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("test_"):
            continue

        test_name = node.name
        has_skip, has_skipif = _extract_decorators(node)

        all_calls = _find_llm_or_vllmrunner_calls(node.body)
        config_keys = _extract_all_kwarg_keys(node.body)
        kwarg_vals = {}
        for call in all_calls:
            for kw in call.keywords:
                if kw.arg and isinstance(kw.value, ast.Constant):
                    kwarg_vals[kw.arg] = kw.value.value
        env_vars = _extract_env_vars(node.body)
        method_calls = _find_method_calls(node.body)
        lora_count = _find_lora_requests(node.body)

        models_from_params = _extract_models_from_parametrize(node, source_code)
        server_models = _extract_server_args_models(node.body, source_code)

        all_models = []
        if models_from_params:
            all_models = list(models_from_params)
        elif server_models:
            all_models = sorted(server_models)
        else:
            found_models = set()
            for call in all_calls:
                m = _extract_model_from_call(call)
                if m:
                    found_models.add(m)
            for call in all_calls:
                if isinstance(call.func, ast.Name) and call.func.id == "compare_logprobs":
                    m = _extract_kwarg_value(call, "model_name")
                    if m:
                        found_models.add(m)
            for var_name in ("model", "model_name", "MODEL_NAME", "DEFAULT_MODEL", "MODEL_PATH"):
                for assign in _walk_body(node.body):
                    if isinstance(assign, ast.Assign):
                        for target in assign.targets:
                            if isinstance(target, ast.Name) and target.id == var_name:
                                val = _source_to_str(assign.value)
                                if val and "/" in val:
                                    found_models.add(val)
            for var_name in ("MODELS", "CROSS_ENCODER_MODELS", "EMBEDDING_MODELS", "WHISPER_MODELS", "MINICPM_MODELS"):
                for assign in ast.walk(tree):
                    if isinstance(assign, ast.Assign):
                        for target in assign.targets:
                            if isinstance(target, ast.Name) and target.id == var_name:
                                if isinstance(assign.value, ast.List):
                                    for elt in assign.value.elts:
                                        v = _source_to_str(elt)
                                        if v and "/" in v:
                                            found_models.add(v)
            if found_models:
                all_models = sorted(found_models)

        if not all_models:
            model_str = "-"
        else:
            model_str = "<br>".join(all_models)

        model_flags = {}
        for m in all_models:
            mf = _classify_model(m)
            for k, v in mf.items():
                if v:
                    model_flags[k] = True

        if not model_flags and not all_models:
            model_flags = {}

        tp = _extract_kwarg_int_from_all(all_calls, "tensor_parallel_size")
        pp = _extract_kwarg_int_from_all(all_calls, "pipeline_parallel_size")
        ep = _extract_kwarg_bool_from_all(all_calls, "enable_expert_parallel")
        pcp = _extract_kwarg_int_from_all(all_calls, "prefill_context_parallel_size")
        dcp = _extract_kwarg_int_from_all(all_calls, "decode_context_parallel_size")
        enforce_eager = _extract_kwarg_bool_from_all(all_calls, "enforce_eager")
        dist_exec_mp = _extract_kwarg_bool_from_all(all_calls, "distributed_executor_backend")

        cudagraph_mode = _extract_cudagraph_mode(all_calls)
        has_cudagraph_sizes = _has_cudagraph_capture_sizes(all_calls)

        enable_lora = _extract_kwarg_bool_from_all(all_calls, "enable_lora")
        max_loras = _extract_kwarg_int_from_all(all_calls, "max_loras")
        fully_sharded = _extract_kwarg_bool_from_all(all_calls, "fully_sharded_loras")

        spec_config = None
        for call in all_calls:
            sc = _extract_kwarg_dict(call, "speculative_config")
            if sc:
                spec_config = sc

        kv_transfer_config = None
        for call in all_calls:
            kc = _extract_kwarg_dict(call, "kv_transfer_config")
            if kc:
                kv_transfer_config = kc

        kv_events_config = None
        for call in all_calls:
            kec = _extract_kwarg_dict(call, "kv_events_config")
            if kec:
                kv_events_config = kec

        runner = _extract_kwarg_value_from_all(all_calls, "runner")
        enable_prefix_caching = _extract_kwarg_bool_from_all(all_calls, "enable_prefix_caching")
        enable_chunked_prefill = _extract_kwarg_bool_from_all(all_calls, "enable_chunked_prefill")
        cp_kv_interleave = _extract_kwarg_int_from_all(all_calls, "cp_kv_cache_interleave_size")
        max_model_len = _extract_kwarg_int_from_all(all_calls, "max_model_len")
        limit_mm = _extract_kwarg_value_from_all(all_calls, "limit_mm_per_prompt")
        attention_backend = _extract_kwarg_value_from_all(all_calls, "attention_backend")
        quantization = _extract_kwarg_value_from_all(all_calls, "quantization")
        dtype = _extract_kwarg_value_from_all(all_calls, "dtype")
        sleep_mode = _extract_kwarg_bool_from_all(all_calls, "enable_sleep_mode")

        add_config = None
        for call in all_calls:
            ac = _extract_kwarg_dict(call, "additional_config")
            if ac:
                add_config = ac

        for call in all_calls:
            cc = _extract_kwarg_dict(call, "compilation_config")
            if cc:
                break

        xlite_graph = None
        for call in all_calls:
            xc = _extract_kwarg_dict(call, "additional_config")
            if xc and "xlite_graph_config" in xc:
                xlite_graph = xc["xlite_graph_config"]

        enable_dsa_cp = None
        multistream_moe = None
        enable_flashcomm1 = None
        if add_config:
            if "enable_dsa_cp" in add_config:
                enable_dsa_cp = add_config["enable_dsa_cp"]
            if "enable_multistream_moe" in add_config:
                multistream_moe = add_config["enable_multistream_moe"]
            if "multistream_overlap_shared_expert" in add_config:
                multistream_moe = add_config["multistream_overlap_shared_expert"]
            if "enable_flashcomm1" in add_config:
                enable_flashcomm1 = add_config["enable_flashcomm1"]

        eplb_config = None
        for call in all_calls:
            ec = _extract_kwarg_dict(call, "additional_config")
            if ec and "eplb_config" in ec:
                eplb_config = ec["eplb_config"]

        has_mamba_ssm_dtype = "mamba_ssm_cache_dtype" in config_keys

        row = {}

        row["Test file"] = display_path
        row["_orig_rel_path"] = rel_path
        row["Test method"] = test_name
        row["Model"] = model_str
        row["310P"] = CHECK if is_310p else EMPTY

        row["Dense"] = CHECK if model_flags.get("Dense") else EMPTY
        row["MoE"] = CHECK if model_flags.get("MoE") else EMPTY
        row["Embedding"] = CHECK if model_flags.get("Embedding") else EMPTY
        row["Classification"] = CHECK if model_flags.get("Classification") else EMPTY
        row["Reranker"] = CHECK if model_flags.get("Reranker") else EMPTY
        row["Mamba/SSM"] = CHECK if (model_flags.get("Mamba/SSM") or has_mamba_ssm_dtype) else EMPTY
        row["Multimodal Reasoning"] = CHECK if (model_flags.get("Multimodal Reasoning") or limit_mm) else EMPTY

        row["TP"] = CHECK if (tp is not None and tp > 1) else EMPTY
        row["PP"] = CHECK if (pp is not None and pp > 1) else EMPTY
        row["EP"] = CHECK if ep else EMPTY
        row["PCP"] = CHECK if (pcp is not None and pcp > 1) else EMPTY
        row["DCP"] = CHECK if (dcp is not None and dcp > 1) else EMPTY
        row["Context Parallel"] = CHECK if ((pcp is not None and pcp > 1) or (dcp is not None and dcp > 1)) else EMPTY

        has_eplb = (
            eplb_config is not None or "expert_parallel_load_balancing" in config_keys or "eplb_config" in config_keys
        )
        row["EPLB"] = CHECK if has_eplb else EMPTY

        has_dynamic_eplb = False
        if eplb_config and isinstance(eplb_config, dict):
            has_dynamic_eplb = eplb_config.get("dynamic_eplb") or "eplb_policy_type" in eplb_config
        if "dynamic_eplb" in config_keys or "DYNAMIC_EPLB" in env_vars:
            has_dynamic_eplb = True
        row["Dynamic EPLB"] = CHECK if has_dynamic_eplb else EMPTY

        row["Multistream MoE"] = CHECK if multistream_moe else EMPTY

        if cudagraph_mode == "FULL":
            row["Full Graph"] = CHECK
        elif cudagraph_mode == "FULL_DECODE_ONLY":
            row["Full Decode Only Graph"] = CHECK
        elif cudagraph_mode == "PIECEWISE":
            row["Piecewise Graph"] = CHECK
        elif cudagraph_mode == "default":
            if enforce_eager:
                row["Eager Mode"] = CHECK
            elif has_cudagraph_sizes:
                row["Default FULL_AND_PIECEWISE Graph"] = CHECK
            else:
                row["Eager Mode"] = CHECK
        elif enforce_eager:
            row["Eager Mode"] = CHECK
        else:
            row["Eager Mode"] = EMPTY

        has_pd_disagg = False
        if kv_transfer_config:
            kv_role = kv_transfer_config.get("kv_role")
            if kv_role and ("kv_producer" in str(kv_role) or "kv_consumer" in str(kv_role)):
                has_pd_disagg = True
        if "pd_disaggregation" in config_keys:
            has_pd_disagg = True
        for node2 in _walk_body(node.body):
            if isinstance(node2, ast.Name) and node2.id in ("RemotePDServer", "DisaggPDProxy"):
                has_pd_disagg = True
        row["PD disaggregation"] = CHECK if has_pd_disagg else EMPTY

        is_w8a8 = any("w8a8" in m.lower() for m in all_models) or (
            quantization == "ascend" and any("w8a8" in m.lower() for m in all_models)
        )
        is_w4a8 = any("w4a8" in m.lower() for m in all_models) or (
            quantization == "ascend" and any("w4a8" in m.lower() for m in all_models)
        )
        row["W8A8"] = CHECK if is_w8a8 else EMPTY
        row["W4A8"] = CHECK if is_w4a8 else EMPTY

        is_fp16 = dtype in ("float16", "half") or (is_310p and dtype != "bfloat16" and not is_w8a8)
        row["FP16"] = CHECK if is_fp16 else EMPTY

        row["LoRA"] = CHECK if enable_lora else EMPTY
        row["Multi-LoRA"] = (
            CHECK if (lora_count > 2 or (max_loras is not None and max_loras > 1 and lora_count >= 2)) else EMPTY
        )

        has_runtime_lora = False
        for mn in method_calls:
            if mn in ("add_lora", "remove_lora", "reload_lora"):
                has_runtime_lora = True
        if "VLLM_ALLOW_RUNTIME_LORA_UPDATING" in env_vars:
            has_runtime_lora = True
        row["Runtime LoRA updating"] = CHECK if has_runtime_lora else EMPTY
        row["Fully sharded LoRA parameterization"] = CHECK if fully_sharded else EMPTY

        has_spec = spec_config is not None or "speculative_config" in config_keys or "speculative_method" in config_keys
        row["Spec Decode"] = CHECK if has_spec else EMPTY

        has_mtp = False
        if spec_config and isinstance(spec_config, dict):
            method = spec_config.get("method", "")
            if method and "mtp" in method.lower():
                has_mtp = True
            if method in ("deepseek_mtp", "qwen3_5_mtp", "qwen3_next_mtp"):
                has_mtp = True
        if "mtp" in test_name.lower():
            has_mtp = True
        row["MTP"] = CHECK if has_mtp else EMPTY

        has_eagle3 = False
        if spec_config and isinstance(spec_config, dict):
            method = spec_config.get("method", "")
            if method and ("eagle3" in method.lower() or method == "eagle"):
                spec_model = spec_config.get("model") or ""
                if "eagle3" in spec_model.lower() or method == "eagle3":
                    has_eagle3 = True
        for m in all_models:
            if "eagle3" in m.lower():
                has_eagle3 = True
        row["Eagle-3"] = CHECK if has_eagle3 else EMPTY

        has_sfa_dsa = False
        if (
            enable_dsa_cp
            or "enable_dsa_cp" in config_keys
            or "sfa" in str(add_config).lower()
            or "dsa" in str(add_config).lower()
        ):
            has_sfa_dsa = True
        if has_sfa_dsa:
            row["SFA/DSA"] = CHECK
        else:
            row["SFA/DSA"] = EMPTY

        has_dsa_cp = enable_dsa_cp and ((pcp is not None and pcp > 1) or (dcp is not None and dcp > 1))
        row["DSA CP"] = CHECK if has_dsa_cp else EMPTY

        row["Pooling runner"] = (
            CHECK if (runner == "pooling" or "task" in config_keys and kwarg_vals.get("task") == "pooling") else EMPTY
        )
        row["Score API"] = CHECK if "score" in method_calls else EMPTY
        row["Classification API"] = CHECK if "classify" in method_calls else EMPTY

        row["Distributed executor mp"] = CHECK if dist_exec_mp else EMPTY

        has_fa3 = False
        if attention_backend and "FLASH_ATTN" in str(attention_backend):
            has_fa3 = True
        if "FA3" in str(attention_backend):
            has_fa3 = True
        if "FLASH_ATTN" in str(attention_backend):
            has_fa3 = True
        row["Flash Attention 3"] = CHECK if has_fa3 else EMPTY

        has_fia = False
        if "FIA" in test_name or "fia" in source_code.lower()[:5000]:
            has_fia = True
        for m in method_calls:
            if "_assert_outputs_match" in m or "_generate_with_backend" in m:
                has_fia = True
        has_fa3_comparison = has_fa3 and has_fia
        row["FIA comparison"] = CHECK if has_fa3_comparison else EMPTY

        row["Chunked Prefill"] = CHECK if (enable_chunked_prefill or "enable_chunked_prefill" in config_keys) else EMPTY
        row["Prefix Caching"] = CHECK if (enable_prefix_caching or "enable_prefix_caching" in config_keys) else EMPTY

        has_cpu_offloading = False
        if "OffloadingConnector" in source_code or "cpu_offloading" in config_keys or "kv_connector" in config_keys:
            for call in all_calls:
                kc = _extract_kwarg_dict(call, "kv_transfer_config")
                if kc and kc.get("kv_connector") == "OffloadingConnector":
                    has_cpu_offloading = True
        row["CPU/KV offloading"] = CHECK if has_cpu_offloading else EMPTY

        has_kv_transfer = False
        if kv_transfer_config:
            connector = kv_transfer_config.get("kv_connector")
            if connector and connector != "OffloadingConnector" and connector != "ExampleHiddenStatesConnector":
                has_kv_transfer = True
        if kv_events_config:
            has_kv_transfer = True
        row["KV transfer/events"] = CHECK if has_kv_transfer else EMPTY

        row["Sleep/Wake memory"] = CHECK if sleep_mode else EMPTY
        row["Xlite Graph"] = CHECK if xlite_graph else EMPTY
        row["CP KV Interleave"] = CHECK if (cp_kv_interleave is not None and cp_kv_interleave > 0) else EMPTY

        is_long_seq = (
            (max_model_len is not None and max_model_len > 8192)
            or "long_sequence" in rel_path
            or "long_sequence" in test_name
        )
        row["Long Sequence"] = CHECK if is_long_seq else EMPTY

        has_flashcomm1 = (
            "VLLM_ASCEND_ENABLE_FLASHCOMM1" in env_vars and env_vars.get("VLLM_ASCEND_ENABLE_FLASHCOMM1") == "1"
        )
        if enable_flashcomm1:
            has_flashcomm1 = True
        row["FlashComm1 env"] = CHECK if has_flashcomm1 else EMPTY

        row["Skipped"] = CHECK if has_skip else EMPTY
        row["Conditional skip"] = CHECK if has_skipif else EMPTY

        has_logprobs = "logprobs" in config_keys or "prompt_logprobs" in config_keys or "num_logprobs" in config_keys
        if "logprobs" in method_calls or "generate_greedy_logprobs" in method_calls:
            has_logprobs = True
        if "compare_logprobs" in method_calls:
            has_logprobs = True
        row["Logprobs"] = CHECK if has_logprobs else EMPTY

        has_batch = False
        for node2 in _walk_body(node.body):
            if isinstance(node2, ast.List):
                if len(node2.elts) > 1:
                    for elt in node2.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            if len(elt.value) > 5:
                                has_batch = True
        if "max_num_seqs" in config_keys:
            mn_seqs = kwarg_vals.get("max_num_seqs")
            if mn_seqs and mn_seqs > 1:
                has_batch = True
        if "batch" in test_name.lower():
            has_batch = True
        row["Batch inference"] = CHECK if has_batch else EMPTY

        has_mixed_lengths = (
            "mixed" in test_name.lower() or "mixed_length" in config_keys or "mixed_lengths" in test_name.lower()
        )
        for node2 in _walk_body(node.body):
            if isinstance(node2, ast.Constant) and isinstance(node2.value, str):
                if "mixed" in node2.value.lower() and "length" in node2.value.lower():
                    has_mixed_lengths = True
        row["Mixed lengths"] = CHECK if has_mixed_lengths else EMPTY

        results.append(row)

    return results


def _extract_kwarg_int_from_all(calls, key):
    for call in calls:
        v = _extract_kwarg_int(call, key)
        if v is not None:
            return v
    return None


def _extract_kwarg_bool_from_all(calls, key):
    for call in calls:
        v = _extract_kwarg_bool(call, key)
        if v is not None:
            return v
    return None


def _extract_kwarg_value_from_all(calls, key):
    for call in calls:
        v = _extract_kwarg_value(call, key)
        if v is not None:
            return v
    return None


def _extract_cudagraph_mode(all_calls):
    for call in all_calls:
        cc = _extract_kwarg_dict(call, "compilation_config")
        if cc:
            mode = cc.get("cudagraph_mode")
            if mode:
                cudagraph_mode_map = {
                    "FULL_DECODE_ONLY": "FULL_DECODE_ONLY",
                    "PIECEWISE": "PIECEWISE",
                    "FULL": "FULL",
                }
                return cudagraph_mode_map[mode]
        cg_mode = _extract_kwarg_value(call, "cudagraph_mode")
        if cg_mode:
            return cg_mode
    return "default"


def _has_cudagraph_capture_sizes(all_calls):
    for call in all_calls:
        cc = _extract_kwarg_dict(call, "compilation_config")
        if cc and "cudagraph_capture_sizes" in cc:
            return True
        if _extract_kwarg_value(call, "cudagraph_capture_sizes") is not None:
            return True
    return False


def _parse_server_cmd(server_cmd):
    features: dict[str, Any] = {}
    if isinstance(server_cmd, str):
        args = server_cmd.split()
    elif isinstance(server_cmd, list):
        args = [str(a) for a in server_cmd]
    else:
        return features

    def _try_json(raw):
        if isinstance(raw, str):
            raw = raw.strip("'").strip('"')
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    JSON_FLAGS = {
        "--compilation-config": "compilation_config",
        "--speculative-config": "speculative_config",
        "--kv-transfer-config": "kv_transfer_config",
        "--additional-config": "additional_config",
    }

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--tensor-parallel-size" and i + 1 < len(args):
            with contextlib.suppress(ValueError):
                features["tp"] = int(args[i + 1])
            i += 2
        elif a == "--pipeline-parallel-size" and i + 1 < len(args):
            with contextlib.suppress(ValueError):
                features["pp"] = int(args[i + 1])
            i += 2
        elif a == "--enforce-eager":
            features["enforce_eager"] = True
            i += 1
        elif a == "--enable-expert-parallel":
            features["ep"] = True
            i += 1
        elif a in ("--quantization", "--dtype", "--attention-backend", "--runner") and i + 1 < len(args):
            features[a.removeprefix("--").replace("-", "_")] = args[i + 1]
            i += 2
        elif a in JSON_FLAGS and i + 1 < len(args):
            val = _try_json(args[i + 1])
            if val is not None:
                features[JSON_FLAGS[a]] = val
            i += 2
        elif a == "--enable-prefix-caching":
            features["enable_prefix_caching"] = True
            i += 1
        elif a == "--no-enable-prefix-caching":
            features["enable_prefix_caching"] = False
            i += 1
        elif a == "--enable-chunked-prefill":
            features["enable_chunked_prefill"] = True
            i += 1
        elif a == "--max-model-len" and i + 1 < len(args):
            with contextlib.suppress(ValueError):
                features["max_model_len"] = int(args[i + 1])
            i += 2
        elif a == "--cp-kv-cache-interleave-size" and i + 1 < len(args):
            with contextlib.suppress(ValueError):
                features["cp_kv_interleave"] = int(args[i + 1])
            i += 2
        elif a == "--enable-sleep-mode":
            features["enable_sleep_mode"] = True
            i += 1
        elif a == "--enable-lora":
            features["enable_lora"] = True
            i += 1
        elif a == "--max-loras" and i + 1 < len(args):
            with contextlib.suppress(ValueError):
                features["max_loras"] = int(args[i + 1])
                pass
            i += 2
        elif a == "--fully-sharded-loras":
            features["fully_sharded_loras"] = True
            i += 1
        elif a == "--distributed-executor-backend" and i + 1 < len(args):
            features["dist_exec_mp"] = True
            i += 2
        else:
            i += 1

    if isinstance(server_cmd, str):
        for flag, key in JSON_FLAGS.items():
            if key not in features:
                m = re.search(rf"{flag}\s+'([^']+)'", server_cmd)
                if m:
                    val = _try_json(m.group(1))
                    if val is not None:
                        features[key] = val

    return features


def _process_yaml_file(filepath, source_code, root_path):
    try:
        import yaml

        data = yaml.safe_load(source_code)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []

    rel_path = str(filepath.relative_to(root_path))
    is_310p = _detect_310p(rel_path)

    display_path = rel_path

    def _build_row(test_name, model, features):
        model_flags = _classify_model(model) if model else {}
        tp = features.get("tp")
        pp = features.get("pp")
        ep = features.get("ep")
        enforce_eager = features.get("enforce_eager")
        dtype = features.get("dtype")
        compilation_config = features.get("compilation_config")
        speculative_config = features.get("speculative_config")
        kv_transfer_config = features.get("kv_transfer_config")
        additional_config = features.get("additional_config")
        enable_prefix_caching = features.get("enable_prefix_caching")
        enable_chunked_prefill = features.get("enable_chunked_prefill")
        cp_kv_interleave = features.get("cp_kv_interleave")
        max_model_len = features.get("max_model_len")
        enable_lora = features.get("enable_lora")
        max_loras = features.get("max_loras")
        fully_sharded = features.get("fully_sharded_loras")
        attention_backend = features.get("attention_backend")
        runner = features.get("runner")
        sleep_mode = features.get("enable_sleep_mode")
        dist_exec_mp = features.get("dist_exec_mp")
        env_vars = features.get("envs", {})

        cudagraph_mode = "default"
        has_cudagraph_sizes = False
        if compilation_config:
            mode = compilation_config.get("cudagraph_mode", "")
            if mode:
                cudagraph_mode = mode
            if "cudagraph_capture_sizes" in compilation_config:
                has_cudagraph_sizes = True

        eplb_config = None
        multistream_moe = None
        enable_dsa_cp = None
        enable_flashcomm1 = None
        if additional_config:
            if "eplb_config" in additional_config:
                eplb_config = additional_config["eplb_config"]
            if "enable_multistream_moe" in additional_config:
                multistream_moe = additional_config["enable_multistream_moe"]
            if "multistream_overlap_shared_expert" in additional_config:
                multistream_moe = additional_config["multistream_overlap_shared_expert"]
            if "enable_dsa_cp" in additional_config:
                enable_dsa_cp = additional_config["enable_dsa_cp"]
            if "enable_flashcomm1" in additional_config:
                enable_flashcomm1 = additional_config["enable_flashcomm1"]
            if "ascend_compilation_config" in additional_config:
                ascend_cc = additional_config["ascend_compilation_config"]
                if ascend_cc.get("enable_npugraph_ex"):
                    has_cudagraph_sizes = True

        row = {}
        row["Test file"] = display_path
        row["_orig_rel_path"] = rel_path
        row["Test method"] = test_name
        row["Model"] = model if model else "-"
        row["310P"] = CHECK if is_310p else EMPTY

        row["Dense"] = CHECK if model_flags.get("Dense") else EMPTY
        row["MoE"] = CHECK if model_flags.get("MoE") else EMPTY
        row["Embedding"] = CHECK if model_flags.get("Embedding") else EMPTY
        row["Classification"] = CHECK if model_flags.get("Classification") else EMPTY
        row["Reranker"] = CHECK if model_flags.get("Reranker") else EMPTY
        row["Mamba/SSM"] = CHECK if model_flags.get("Mamba/SSM") else EMPTY
        row["Multimodal Reasoning"] = CHECK if model_flags.get("Multimodal Reasoning") else EMPTY

        row["TP"] = CHECK if (tp is not None and tp > 1) else EMPTY
        row["PP"] = CHECK if (pp is not None and pp > 1) else EMPTY
        row["EP"] = CHECK if ep else EMPTY
        row["PCP"] = EMPTY
        row["DCP"] = EMPTY
        row["Context Parallel"] = EMPTY
        row["EPLB"] = CHECK if eplb_config else EMPTY
        row["Dynamic EPLB"] = EMPTY
        row["Multistream MoE"] = CHECK if multistream_moe else EMPTY

        if cudagraph_mode == "FULL":
            row["Full Graph"] = CHECK
        elif cudagraph_mode == "FULL_DECODE_ONLY":
            row["Full Decode Only Graph"] = CHECK
        elif cudagraph_mode == "PIECEWISE":
            row["Piecewise Graph"] = CHECK
        elif cudagraph_mode == "default":
            if enforce_eager:
                row["Eager Mode"] = CHECK
            elif has_cudagraph_sizes:
                row["Default FULL_AND_PIECEWISE Graph"] = CHECK
            else:
                row["Eager Mode"] = CHECK
        elif enforce_eager:
            row["Eager Mode"] = CHECK
        else:
            row["Eager Mode"] = EMPTY

        has_pd_disagg = False
        if kv_transfer_config:
            kv_role = kv_transfer_config.get("kv_role", "")
            if "kv_producer" in str(kv_role) or "kv_consumer" in str(kv_role):
                has_pd_disagg = True
        row["PD disaggregation"] = CHECK if has_pd_disagg else EMPTY

        is_w8a8 = "w8a8" in model.lower()
        is_w4a8 = "w4a8" in model.lower()
        row["W8A8"] = CHECK if is_w8a8 else EMPTY
        row["W4A8"] = CHECK if is_w4a8 else EMPTY

        is_fp16 = dtype in ("float16", "half") or (is_310p and dtype != "bfloat16" and not is_w8a8)
        row["FP16"] = CHECK if is_fp16 else EMPTY

        row["LoRA"] = CHECK if enable_lora else EMPTY
        row["Multi-LoRA"] = CHECK if (max_loras is not None and max_loras > 1) else EMPTY
        row["Runtime LoRA updating"] = EMPTY
        row["Fully sharded LoRA parameterization"] = CHECK if fully_sharded else EMPTY

        has_spec = speculative_config is not None
        row["Spec Decode"] = CHECK if has_spec else EMPTY

        has_mtp = False
        if speculative_config and isinstance(speculative_config, dict):
            method = speculative_config.get("method", "")
            if method and "mtp" in method.lower():
                has_mtp = True
        row["MTP"] = CHECK if has_mtp else EMPTY

        has_eagle3 = False
        if speculative_config and isinstance(speculative_config, dict):
            method = speculative_config.get("method", "")
            if method and "eagle3" in method.lower():
                has_eagle3 = True
        row["Eagle-3"] = CHECK if has_eagle3 else EMPTY

        has_sfa_dsa = bool(enable_dsa_cp)
        if has_sfa_dsa:
            row["SFA/DSA"] = CHECK
        else:
            row["SFA/DSA"] = EMPTY
        row["DSA CP"] = EMPTY

        row["Pooling runner"] = CHECK if runner == "pooling" else EMPTY
        row["Score API"] = EMPTY
        row["Classification API"] = EMPTY
        row["Distributed executor mp"] = CHECK if dist_exec_mp else EMPTY

        has_fa3 = attention_backend and "FLASH_ATTN" in str(attention_backend)
        row["Flash Attention 3"] = CHECK if has_fa3 else EMPTY
        row["FIA comparison"] = EMPTY

        row["Chunked Prefill"] = CHECK if enable_chunked_prefill else EMPTY
        row["Prefix Caching"] = CHECK if enable_prefix_caching else EMPTY

        row["CPU/KV offloading"] = EMPTY

        has_kv_transfer = False
        if kv_transfer_config:
            connector = kv_transfer_config.get("kv_connector", "")
            if connector and connector != "OffloadingConnector" and connector != "ExampleHiddenStatesConnector":
                has_kv_transfer = True
        row["KV transfer/events"] = CHECK if has_kv_transfer else EMPTY

        row["Sleep/Wake memory"] = CHECK if sleep_mode else EMPTY

        has_xlite = False
        if additional_config and "xlite_graph_config" in additional_config:
            has_xlite = True
        row["Xlite Graph"] = CHECK if has_xlite else EMPTY

        row["CP KV Interleave"] = CHECK if (cp_kv_interleave is not None and cp_kv_interleave > 0) else EMPTY

        is_long_seq = max_model_len is not None and max_model_len > 8192
        row["Long Sequence"] = CHECK if is_long_seq else EMPTY

        has_flashcomm1 = env_vars.get("VLLM_ASCEND_ENABLE_FLASHCOMM1") == "1"
        if enable_flashcomm1:
            has_flashcomm1 = True
        row["FlashComm1 env"] = CHECK if has_flashcomm1 else EMPTY

        row["Skipped"] = EMPTY
        row["Conditional skip"] = EMPTY
        row["Logprobs"] = EMPTY
        row["Batch inference"] = EMPTY
        row["Mixed lengths"] = EMPTY

        return row

    if "test_name" in data and "model" in data:
        envs = dict(data.get("env_common", {}) or data.get("envs", {}))
        features: dict[str, Any] = {}
        deployment = data.get("deployment", [])
        if deployment:
            features.update(_parse_server_cmd(deployment[0].get("server_cmd", "")))
            if "envs" in deployment[0]:
                envs.update(deployment[0]["envs"])
        templates = data.get("templates", [])
        if templates:
            features.update(_parse_server_cmd(templates[0].get("server_cmd_template", [])))
            if "envs" in templates[0]:
                envs.update(templates[0]["envs"])
        features["envs"] = envs
        row = _build_row(data["test_name"], data.get("model", ""), features)
        results = [row]

    elif "test_cases" in data:
        results = []
        for tc in data["test_cases"]:
            features = {}
            server_cmd = tc.get("server_cmd", [])
            if server_cmd:
                features.update(_parse_server_cmd(server_cmd))
            features["envs"] = tc.get("envs", {})
            row = _build_row(tc.get("name", tc.get("model", "")), tc.get("model", ""), features)
            results.append(row)
    else:
        return []

    return results


def main():
    rows = []
    test_files = sorted(E2E_PR_ROOT.rglob("test_*.py"))

    for filepath in test_files:
        source_code = filepath.read_text(encoding="utf-8", errors="replace")
        file_rows = _process_test_file(filepath, source_code)
        rows.extend(file_rows)

    rows.sort(key=lambda r: (r["Test file"], r["Test method"]))

    card_sections_rows: dict[str, list] = {}
    for section_title, card_prefix in CARD_SECTIONS:
        card_sections_rows[section_title] = []
    for row in rows:
        orig_path = row.get("_orig_rel_path", "")
        for section_title, card_prefix in CARD_SECTIONS:
            if orig_path.startswith(card_prefix + "/"):
                card_sections_rows[section_title].append(row)
                break

    NIGHTLY_ROOT = REPO_ROOT / "tests" / "e2e" / "nightly"
    WEEKLY_ROOT = REPO_ROOT / "tests" / "e2e" / "weekly"

    def _scan_tests(root):
        py_rows = []
        yaml_rows = []
        for fp in sorted(root.rglob("test_*.py")):
            src = fp.read_text(encoding="utf-8", errors="replace")
            py_rows.extend(_process_test_file(fp, src, root))
        for fp in sorted(root.rglob("*.yaml")):
            src = fp.read_text(encoding="utf-8", errors="replace")
            yaml_rows.extend(_process_yaml_file(fp, src, root))
        combined = py_rows + yaml_rows
        combined.sort(key=lambda r: (r["Test file"], r["Test method"]))
        return combined

    nightly_rows = _scan_tests(NIGHTLY_ROOT)
    weekly_rows = _scan_tests(WEEKLY_ROOT)

    header = "| " + " | ".join(COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(COLUMNS)) + " |"

    output = "The coverage of e2e is as follows:\n\n"
    for section_title, card_prefix in CARD_SECTIONS:
        section_rows = card_sections_rows[section_title]
        output += f"## {section_title}\n\n"
        output += header + "\n"
        output += separator + "\n"
        for row in section_rows:
            vals = [row.get(col, EMPTY) for col in COLUMNS]
            output += "| " + " | ".join(vals) + " |\n"
        output += "\n"

    for section_title, rows_data in [("Nightly Tests", nightly_rows), ("Weekly Tests", weekly_rows)]:
        output += f"## {section_title}\n\n"
        output += header + "\n"
        output += separator + "\n"
        for row in rows_data:
            vals = [row.get(col, EMPTY) for col in COLUMNS]
            output += "| " + " | ".join(vals) + " |\n"
        output += "\n"

    print(output)

    OUTPUT_FILE.write_text(output.rstrip("\n") + "\n", encoding="utf-8")
    print(f"\nWritten to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
