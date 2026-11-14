from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import torch
import torch_npu
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.model_executor.models.qwen3 import Qwen3DecoderLayer
    from vllm.sequence import IntermediateTensors

BLOCK_SIZE = 128
FA_QBLOCK_SIZE = 128
FA_BC = 1024


logger = logging.getLogger(__name__)


def _force_eager(vllm_config: VllmConfig) -> None:
    from vllm.config import CompilationMode, CUDAGraphMode

    compilation_config = vllm_config.compilation_config
    model_config = vllm_config.model_config
    if (
        getattr(model_config, "enforce_eager", False)
        and compilation_config.mode == CompilationMode.NONE
        and compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    ):
        return

    model_config.enforce_eager = True
    compilation_config.mode = CompilationMode.NONE
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    compilation_config.max_cudagraph_capture_size = 0
    compilation_config.cudagraph_capture_sizes = []
    logger.warning(
        "Qwen3 DBF 融合算子已启用：强制 eager 执行"
        "（compilation mode=NONE, cudagraph_mode=NONE），"
        "命令行中的编译/图模式设置被覆盖。"
    )


def get_attn_metadata():
    from vllm.forward_context import get_forward_context

    attn_metadata = get_forward_context().attn_metadata
    if isinstance(attn_metadata, dict):
        for v in attn_metadata.values():
            attn_metadata = v
            break
    return attn_metadata


class Task:
    def __init__(self, req_id: int, q_id: int, cost: int):
        self.req_id = req_id
        self.q_id = q_id
        self.task_id = (req_id << 16) | q_id
        self.cost = cost


def lpt_split(tasks: list[Task], p: int, *, device: torch.device):
    if not tasks:
        return torch.empty((p, 0), dtype=torch.int32, device=device)

    pkgs = [[] for _ in range(p)]
    load = [0] * p
    for t in sorted(tasks, key=lambda x: -x.cost):
        k = 0
        for i in range(1, p):
            if (load[i], len(pkgs[i])) < (load[k], len(pkgs[k])):
                k = i
        pkgs[k].append(t.task_id)
        load[k] += t.cost
    for b in pkgs:
        b.sort()
    width = max(len(b) for b in pkgs)
    rows = [b + [-1] * (width - len(b)) for b in pkgs]
    return torch.tensor(rows, dtype=torch.int32, device=device)


def build_tasks(cu_q_lens: list[int], kv_lens: list[int], group: int, head_dim: int, *, device: torch.device):
    def l2_bytes(row: int, cuBr: int):
        gbr = row * group
        q_bytes = gbr * head_dim * 2 * 2
        kv_bytes = cuBr * head_dim * 2 * 2
        sp_bytes = gbr * cuBr * 2 * 6
        bc_blocks = (cuBr + FA_BC - 1) // FA_BC
        if bc_blocks == 1:
            o_bytes = gbr * head_dim * 4 * 2
        else:
            o_bytes = gbr * head_dim * 4 * 4 * (bc_blocks - 1)
        return q_bytes + kv_bytes + sp_bytes + o_bytes

    tasks = []
    for req in range(len(kv_lens)):
        q_len = cu_q_lens[req] - (0 if req == 0 else cu_q_lens[req - 1])
        chunked = kv_lens[req] - q_len
        row = min(FA_QBLOCK_SIZE - (chunked % FA_QBLOCK_SIZE), q_len)
        cost = l2_bytes(row, chunked + row)
        tasks.append(Task(req, 0, cost))
        q = 1
        r = row
        while r < q_len:
            row = min(FA_QBLOCK_SIZE, q_len - r)
            cost = l2_bytes(row, chunked + r + row)
            tasks.append(Task(req, q, cost))
            q += 1
            r += row
    return lpt_split(tasks, p=torch.npu.get_device_properties().cube_core_num // 4, device=device)


def split_double_batch(attn_metadata, num_reqs, *, device: torch.device):
    actual_seq_lengths_q = attn_metadata.actual_seq_lengths_q  # list
    assert attn_metadata.num_actual_tokens == actual_seq_lengths_q[-1]
    left = 0
    right = num_reqs - 1
    num_reqs_0 = -1
    while left < right - 1:
        mid = (left + right) // 2
        num_tokens = 2 * actual_seq_lengths_q[mid]
        if num_tokens == attn_metadata.num_actual_tokens:
            num_reqs_0 = mid + 1
            break
        if num_tokens < attn_metadata.num_actual_tokens:
            left = mid
        else:
            right = mid
    if num_reqs_0 == -1:
        if actual_seq_lengths_q[left] + actual_seq_lengths_q[right] >= attn_metadata.num_actual_tokens:
            num_reqs_0 = left + 1
        else:
            num_reqs_0 = right + 1
    num_tokens_0 = actual_seq_lengths_q[num_reqs_0 - 1]
    num_tokens_1 = actual_seq_lengths_q[-1] - num_tokens_0
    cu_q_lens_0 = torch.tensor(attn_metadata.actual_seq_lengths_q[:num_reqs_0], dtype=torch.int32, device=device)
    cu_q_lens_1 = torch.tensor(
        attn_metadata.actual_seq_lengths_q[num_reqs_0:num_reqs], dtype=torch.int32, device=device
    )
    cu_q_lens_1 -= num_tokens_0
    return num_reqs_0, num_tokens_0, cu_q_lens_0, num_reqs - num_reqs_0, num_tokens_1, cu_q_lens_1


class Qwen3DecodeLayerWeights:
    def __init__(self, decode_layer: Qwen3DecoderLayer):
        # input_layernorm
        self.input_norm_gamma_fp32 = decode_layer.input_layernorm.weight.contiguous().to(dtype=torch.float32)
        self.input_norm_epsilon = decode_layer.input_layernorm.variance_epsilon
        # attn
        self.qkv_weight = torch_npu.npu_format_cast(
            decode_layer.self_attn.qkv_proj.weight.data.t().contiguous(), torch_npu.Format.FRACTAL_NZ
        )
        self.qnorm_gamma_fp32 = decode_layer.self_attn.q_norm.weight.contiguous().to(dtype=torch.float32)
        self.knorm_gamma_fp32 = decode_layer.self_attn.k_norm.weight.contiguous().to(dtype=torch.float32)
        self.knorm_epsilon = decode_layer.self_attn.k_norm.variance_epsilon
        self.o_weight = torch_npu.npu_format_cast(
            decode_layer.self_attn.o_proj.weight.data.t().contiguous(), torch_npu.Format.FRACTAL_NZ
        )
        # mlp
        self.gate_up_weight = torch_npu.npu_format_cast(
            decode_layer.mlp.gate_up_proj.weight.data.t().contiguous(), torch_npu.Format.FRACTAL_NZ
        )
        self.down_weight = torch_npu.npu_format_cast(
            decode_layer.mlp.down_proj.weight.data.t().contiguous(), torch_npu.Format.FRACTAL_NZ
        )
        # post_attention_layernorm
        self.post_norm_gamma_fp32 = decode_layer.post_attention_layernorm.weight.contiguous().to(dtype=torch.float32)
        self.post_norm_epsilon = decode_layer.post_attention_layernorm.variance_epsilon

        del decode_layer.input_layernorm
        del decode_layer.self_attn.qkv_proj
        del decode_layer.self_attn.q_norm
        del decode_layer.self_attn.k_norm
        del decode_layer.self_attn.o_proj
        del decode_layer.mlp
        del decode_layer.post_attention_layernorm


class AscendQwen3DBF:
    def __init__(self, qwen3: Qwen3ForCausalLM, MAX_TOKENS: int):
        import ascend_ops  # noqa: F401

        self.qwen3 = qwen3
        self.kv_caches_binded = False
        # KV cache block 步长（元素数）。MRV1 是 dense（== BLOCK_SIZE*KV_HEADS*HEAD_DIM），
        # MRV2 是页式合并布局（K/V 同页交错，通常 2 倍，还可能带对齐 padding）。
        # 首次 bind 时从真实张量 stride 读取，绝不写死。
        self.kv_block_stride = None

        self.decode_layer_weights = []
        for decode_layer in qwen3.model.layers:
            self.decode_layer_weights.append(Qwen3DecodeLayerWeights(decode_layer))
        self.device = self.decode_layer_weights[0].qkv_weight.device
        self.scaling = qwen3.model.layers[0].self_attn.scaling
        self.q_size = qwen3.model.layers[0].self_attn.q_size
        self.kv_size = qwen3.model.layers[0].self_attn.kv_size
        self.head_dim = qwen3.model.layers[0].self_attn.head_dim
        self.q_heads = self.q_size // self.head_dim
        self.kv_heads = self.kv_size // self.head_dim
        self.group = self.q_heads // self.kv_heads
        assert self.kv_heads == 8 and self.head_dim == 128 and self.group in [4, 5]
        self.cos_sin_cache = qwen3.model.layers[0].self_attn.rotary_emb.cos_sin_cache

        self.norm_gamma_fp32 = qwen3.model.norm.weight.contiguous().to(dtype=torch.float32)
        self.norm_epsilon = qwen3.model.norm.variance_epsilon
        del qwen3.model.norm
        # workspace
        self.hidden_dim = self.norm_gamma_fp32.shape[0]
        dt = self.decode_layer_weights[0].qkv_weight.dtype
        self.hidden_states = torch.empty(MAX_TOKENS, self.hidden_dim, dtype=dt, device=self.device)
        self.residual = torch.empty(MAX_TOKENS, self.hidden_dim, dtype=dt, device=self.device)
        self.qkv = torch.empty(MAX_TOKENS, self.q_size + 2 * self.kv_size, dtype=dt, device=self.device)
        self.q = torch.empty(MAX_TOKENS, self.q_size, dtype=dt, device=self.device)
        self.gate_up_dim = self.decode_layer_weights[0].gate_up_weight.shape[1]
        self.gate_up = torch.empty(MAX_TOKENS, self.gate_up_dim, dtype=dt, device=self.device)
        self.silu = torch.empty(MAX_TOKENS, self.gate_up_dim // 2, dtype=dt, device=self.device)

        self.mask32 = torch.triu(torch.ones(32, 32, dtype=torch.float32, device=self.device), diagonal=1) * (-1e30)
        self.fa_ws = torch.empty(
            torch.npu.get_device_properties().cube_core_num
            * self.group
            * FA_QBLOCK_SIZE
            * (2 * FA_BC + 4 * self.head_dim)
            * 4,
            dtype=torch.uint8,
            device=self.device,
        )

    def bind_kv_caches(self):
        if not self.kv_caches_binded:
            weights_list = []
            self.layer_num = len(self.decode_layer_weights)
            for i in range(self.layer_num):
                weights = self.decode_layer_weights[i]
                attn = self.qwen3.model.layers[i].self_attn.attn
                weights.key_cache = attn.kv_cache[0]
                weights.value_cache = attn.kv_cache[1]
                k_stride = weights.key_cache.stride(0)
                v_stride = weights.value_cache.stride(0)
                assert k_stride == v_stride, f"layer {i}: K/V block stride 不一致 ({k_stride} vs {v_stride})"
                if i == 0:
                    self.kv_block_stride = k_stride
                else:
                    assert self.kv_block_stride == k_stride, (
                        f"layer {i}: KV block stride {k_stride} 与首层 {self.kv_block_stride} 不一致"
                    )
                weights_list.append(weights.input_norm_gamma_fp32.data_ptr())
                weights_list.append(weights.qkv_weight.data_ptr())
                weights_list.append(weights.qnorm_gamma_fp32.data_ptr())
                weights_list.append(weights.knorm_gamma_fp32.data_ptr())
                weights_list.append(weights.key_cache.data_ptr())
                weights_list.append(weights.value_cache.data_ptr())
                weights_list.append(weights.o_weight.data_ptr())
                weights_list.append(weights.post_norm_gamma_fp32.data_ptr())
                weights_list.append(weights.gate_up_weight.data_ptr())
                weights_list.append(weights.down_weight.data_ptr())
            weights_list.append(self.norm_gamma_fp32.data_ptr())
            self.weights_params = torch.tensor(weights_list, dtype=torch.int64, device=self.device)
            self.num_block, self.block_size, _, _ = self.decode_layer_weights[0].key_cache.shape
            assert self.block_size == BLOCK_SIZE
            self.kv_caches_binded = True

    def __forward_naive(self, positions: torch.Tensor, attn_metadata) -> torch.Tensor:
        num_reqs = len(attn_metadata.actual_seq_lengths_q)
        num_tokens = attn_metadata.num_actual_tokens
        kv_lens = torch.tensor(attn_metadata.seq_lens_list[:num_reqs], dtype=torch.int32, device=self.device)
        cu_q_lens = torch.tensor(attn_metadata.actual_seq_lengths_q[:num_reqs], dtype=torch.int32, device=self.device)
        tasks = build_tasks(
            attn_metadata.actual_seq_lengths_q[:num_reqs],
            attn_metadata.seq_lens_list[:num_reqs],
            self.group,
            self.head_dim,
            device=self.device,
        )

        self.residual[:num_tokens] = 0.0
        torch.ops.ascend_ops.qwen3_4b_14b_910b(
            num_tokens,
            self.weights_params,
            self.layer_num,
            self.cos_sin_cache,
            self.hidden_dim,
            self.q_size,
            self.gate_up_dim,
            self.scaling,
            self.kv_block_stride,
            positions,
            attn_metadata.slot_mapping[:num_tokens],
            tasks,
            cu_q_lens,
            kv_lens,
            attn_metadata.block_tables[:num_reqs],
            self.mask32,
            self.fa_ws,
            self.hidden_states,
            self.residual,
            self.qkv,
            self.q,
            self.gate_up,
            self.silu,
        )
        return self.hidden_states[:num_tokens]

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # hidden_states = self.qwen3.model.embed_tokens(input_ids)
        torch.index_select(
            self.qwen3.model.embed_tokens.weight, 0, input_ids, out=self.hidden_states[: input_ids.shape[0]]
        )
        positions = positions.to(dtype=torch.int32)
        attn_metadata = get_attn_metadata()
        if attn_metadata is None:  # dummpy_run
            return self.hidden_states[: input_ids.shape[0]]
        self.bind_kv_caches()
        num_reqs = len(attn_metadata.actual_seq_lengths_q)
        if num_reqs < 2 or attn_metadata.num_actual_tokens <= 1280:
            return self.__forward_naive(positions, attn_metadata)
        num_reqs_0, num_tokens_0, cu_q_lens_0, num_reqs_1, num_tokens_1, cu_q_lens_1 = split_double_batch(
            attn_metadata, num_reqs, device=self.device
        )
        num_tokens = num_tokens_0 + num_tokens_1
        tasks_0 = build_tasks(
            cu_q_lens_0.tolist(),
            attn_metadata.seq_lens_list[:num_reqs_0],
            self.group,
            self.head_dim,
            device=self.device,
        )
        tasks_1 = build_tasks(
            cu_q_lens_1.tolist(),
            attn_metadata.seq_lens_list[num_reqs_0:num_reqs],
            self.group,
            self.head_dim,
            device=self.device,
        )
        kv_lens = torch.tensor(attn_metadata.seq_lens_list[:num_reqs], dtype=torch.int32, device=self.device)

        self.residual[:num_tokens] = 0.0
        torch.ops.ascend_ops.qwen3_4b_14b_910b_dbf(
            num_reqs_0,
            num_tokens_0,
            num_tokens_1,
            self.weights_params,
            self.layer_num,
            self.cos_sin_cache,
            self.hidden_dim,
            self.q_size,
            self.gate_up_dim,
            self.scaling,
            self.kv_block_stride,
            positions,
            attn_metadata.slot_mapping[:num_tokens],
            tasks_0,
            cu_q_lens_0,
            tasks_1,
            cu_q_lens_1,
            kv_lens,
            attn_metadata.block_tables[:num_reqs],
            self.mask32,
            self.fa_ws,
            self.hidden_states,
            self.residual,
            self.qkv,
            self.q,
            self.gate_up,
            self.silu,
        )
        return self.hidden_states[: num_tokens_0 + num_tokens_1]


class AscendQwen3CustomForCausalLM(Qwen3ForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        assert torch.npu.get_device_name().startswith("Ascend910B")
        # 检测到本模型定义：强制 eager，覆盖命令行的编译/图模式设置
        _force_eager(vllm_config)
        # 先调用父类初始化，创建标准模型结构
        # 注意：不在这里初始化dbf，因为权重还未加载
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # 延迟初始化dbf，在权重加载完成后初始化
        self.dbf: AscendQwen3DBF | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input IDs using the embedding table."""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        # 仅支持单卡（PP=1）：不做 intermediate_tensors / 非 last rank 回退。
        if self.dbf is None:
            raise RuntimeError("DBF wrapper not initialized; load_weights() must run before forward()")
        return self.dbf.forward(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """
        Load weights from HuggingFace checkpoint.

        Uses the parent class's weight loading mechanism to ensure correct
        weight mapping, then initializes the DBF wrapper which extracts
        weights into FRACTAL_NZ format for custom operator execution.
        """
        # 使用父类的权重加载机制确保权重正确映射
        loaded_params = super().load_weights(weights)

        # 权重加载完成后，初始化DBF包装器
        # DBF会提取权重到FRACTAL_NZ格式并准备自定义算子资源
        if self.dbf is None:
            max_num_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            self.dbf = AscendQwen3DBF(self, MAX_TOKENS=max_num_tokens)

        return loaded_params
