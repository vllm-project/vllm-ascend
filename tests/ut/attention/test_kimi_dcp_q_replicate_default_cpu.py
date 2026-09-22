# SPDX-License-Identifier: Apache-2.0

import ast
from copy import copy
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3] / "vllm_ascend"


@dataclass
class Parallel:
    tensor_parallel_size: int = 8
    decode_context_parallel_size: int = 8
    prefill_context_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    dcp_q_replicate: bool = False


def configure(config, flash=True, a5=True):
    path = ROOT / "platform.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    helper = next(n for n in tree.body if getattr(n, "name", None) == "_enable_kimi_k3_flash_dcp_q_replicate")
    scope = dict(
        envs=SimpleNamespace(VLLM_ASCEND_ENABLE_FLASH_MLA=flash),
        get_current_hardware_profile=lambda: SimpleNamespace(supports=lambda _: a5),
        HardwareCapability=SimpleNamespace(MLA_DECODE_PROLOG_WITHOUT_ROPE=1),
    )
    source = "from __future__ import annotations\n" + ast.unparse(ast.Module(body=[helper], type_ignores=[]))
    exec(compile(source, str(path), "exec"), scope)
    scope[helper.name](config)


def config():
    return SimpleNamespace(
        model_config=SimpleNamespace(
            use_mla=True, hf_config=SimpleNamespace(architectures=["KimiK3ForConditionalGeneration"])
        ),
        parallel_config=Parallel(),
    )


@pytest.mark.parametrize("runner", ["mrv1", "mrv2"])
def test_default_target_and_mla_dspark_use_group_q_projection(runner):
    cfg = config()
    configure(cfg)
    assert cfg.parallel_config.dcp_q_replicate
    # Both DSpark loaders copy target parallel settings, only overriding
    # execution TP/PP; GQA additionally sets its own DCP size to one.
    draft = replace(cfg.parallel_config, pipeline_parallel_size=1, tensor_parallel_size=8)
    assert draft.dcp_q_replicate and draft.decode_context_parallel_size == 8

    class Upstream:
        def __init__(self, **kwargs):
            self._attention_layer = SimpleNamespace(impl=SimpleNamespace(enable_mlapo=True))

    class GroupLinear:
        def __init__(self, input_size, output_size, **kwargs):
            parallel = cfg.parallel_config
            self.output_size_per_partition = output_size // (
                parallel.tensor_parallel_size // parallel.decode_context_parallel_size
            )

    path = ROOT / "models/kimi_k3.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AscendKimiMLAAttention")
    cls.bases = [ast.copy_location(ast.Name(id="Upstream", ctx=ast.Load()), cls.bases[0])]
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"]
    scope = dict(
        Upstream=Upstream,
        copy=copy,
        get_current_vllm_config=lambda: cfg,
        envs=SimpleNamespace(is_set=lambda _: False),
        DCPGroupColumnParallelLinear=GroupLinear,
        mark_fused_preprocess_weights=lambda _: None,
    )
    exec("from __future__ import annotations\n" + ast.unparse(ast.Module(body=[cls], type_ignores=[])), scope)
    kwargs = dict(
        config=SimpleNamespace(),
        hidden_size=7168,
        num_heads=96,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        use_output_gate=True,
        use_rope=False,
    )
    target = scope[cls.name](**kwargs)
    assert target.q_b_proj.output_size_per_partition == 96 * 192
    assert target._attention_layer.impl.q_proj is target.q_b_proj
    assert target._attention_layer.impl.enable_mlapo
    cfg.parallel_config = draft
    draft_layer = scope[cls.name](**kwargs, disable_mlapo=True)
    assert draft_layer.q_b_proj.output_size_per_partition == 96 * 192
    assert not draft_layer._attention_layer.impl.enable_mlapo


@pytest.mark.parametrize("different", ["flash", "a5", "model", "mla", "tp", "dcp", "pcp"])
def test_other_paths_preserve_q_replication_setting(different):
    cfg = config()
    if different == "model":
        cfg.model_config.hf_config.architectures = ["DeepseekV3ForCausalLM"]
    elif different == "mla":
        cfg.model_config.use_mla = False
    elif different == "tp":
        cfg.parallel_config.tensor_parallel_size = 16
    elif different == "dcp":
        cfg.parallel_config.decode_context_parallel_size = 1
    elif different == "pcp":
        cfg.parallel_config.prefill_context_parallel_size = 2
    configure(cfg, flash=different != "flash", a5=different != "a5")
    assert not cfg.parallel_config.dcp_q_replicate
