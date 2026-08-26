# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import math
import sys
from collections.abc import Iterable
from copy import copy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import vllm
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.dots_ocr import (
    DotsPatchEmbed,
    DotsSwiGLUFFN,
    PatchMerger,
    VisionRotaryEmbedding,
)
from vllm.model_executor.models.dots_ocr import (
    DotsVisionAttention as DotsOCRVisionAttention,
)
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.qwen2_vl import Qwen2VisionAttention
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils.import_utils import import_from_path

from .audio import (
    Dots3NoteAudioTower,
    prepare_audio_features,
)
from .model import Dots3NoteLanguageModelForCausalLM


def _load_vllm_dots3_note_common_module():
    if vllm.__file__ is None:
        raise ImportError("Unable to locate the installed vLLM package")
    common_path = Path(vllm.__file__).resolve().parent / "models" / "dots3_note" / "common"
    if not common_path.is_dir():
        raise ImportError(f"The vLLM Dots3 Note common source was not found at {common_path}")
    package_name = "vllm_ascend.models.dots3_note._vllm_dots3_note_common"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(common_path)]
        sys.modules[package_name] = package
    processor_name = f"{package_name}.processor"
    if processor_name not in sys.modules:
        import_from_path(processor_name, common_path / "processor.py")
    return sys.modules[processor_name]


_common = _load_vllm_dots3_note_common_module()
AUDIO_END = _common.AUDIO_END
AUDIO_PAD = _common.AUDIO_PAD
AUDIO_START = _common.AUDIO_START
IMAGE_END = _common.IMAGE_END
IMAGE_PAD = _common.IMAGE_PAD
IMAGE_START = _common.IMAGE_START
VIDEO_PLACEHOLDER = _common.VIDEO_PLACEHOLDER
Dots3NoteDummyInputsBuilder = _common.Dots3NoteDummyInputsBuilder
Dots3NoteMultiModalProcessor = _common.Dots3NoteMultiModalProcessor
Dots3NoteProcessingInfo = _common.Dots3NoteProcessingInfo
load_note_config_section = _common.load_note_config_section


class DotsVisionAttention(DotsOCRVisionAttention):
    def __init__(
        self,
        config: SimpleNamespace,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config,
            config.embed_dim,
            num_heads=config.num_attention_heads,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.q_norm = (
            RMSNorm(self.hidden_size_per_attention_head, eps=config.rms_norm_eps) if config.use_qk_norm else None
        )
        self.k_norm = (
            RMSNorm(self.hidden_size_per_attention_head, eps=config.rms_norm_eps) if config.use_qk_norm else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        x = hidden_states.unsqueeze(1)
        qkv, _ = self.qkv(x)
        q, k, v = Qwen2VisionAttention.split_qkv(self, qkv)
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        batch_size = q.shape[1]
        q = q.permute(1, 0, 2, 3).contiguous()
        k = k.permute(1, 0, 2, 3).contiguous()
        v = v.permute(1, 0, 2, 3).contiguous()

        qk_concat = torch.cat([q, k], dim=0)
        qk_rotated = self.apply_rotary_emb(
            qk_concat,
            rotary_pos_emb.cos(),
            rotary_pos_emb.sin(),
        )
        q, k = torch.chunk(qk_rotated, 2, dim=0)

        context_layer = self.attn(
            query=q,
            key=k,
            value=v,
            cu_seqlens=cu_seqlens,
        )
        context_layer = context_layer.permute(1, 0, 2, 3).contiguous()
        context_layer = context_layer.view(context_layer.shape[0], batch_size, -1)
        out, _ = self.proj(context_layer)
        return out.squeeze(1)


def _dots_swiglu(
    config: SimpleNamespace,
    intermediate_size: int,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> DotsSwiGLUFFN:
    config = copy(config)
    config.intermediate_size = intermediate_size
    return DotsSwiGLUFFN(config, quant_config=quant_config, prefix=prefix)


class MoESwiGLUFFN(nn.Module):
    def __init__(
        self,
        config: SimpleNamespace,
        layer_number: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_routed = config.pyramid_num_routed[layer_number]
        self.capacity_factor = config.capacity_factor
        self.router_scoring_func = config.router_scoring_func
        self.router_scale = config.router_scale
        self.register_buffer(
            "router_bias",
            torch.zeros(self.num_routed, dtype=torch.float32),
        )
        self.experts = nn.ModuleList(
            [
                _dots_swiglu(
                    config,
                    config.moe_intermediate_size,
                    quant_config,
                    f"{prefix}.experts.{expert_idx}",
                )
                for expert_idx in range(self.num_routed)
            ]
        )
        self.gate_weight = nn.Parameter(torch.empty((self.num_routed, config.embed_dim), dtype=torch.float32))
        nn.init.kaiming_uniform_(self.gate_weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        epsilon = 1e-9
        x_flat = x.contiguous().view(-1, x.shape[-1])
        num_tokens = x_flat.shape[0]
        gate_logits = F.linear(x_flat.float(), self.gate_weight.float())
        if self.router_scoring_func == "sigmoid":
            gating_prob = torch.sigmoid(gate_logits)
        else:
            gating_prob = torch.softmax(gate_logits, dim=-1, dtype=torch.float32)

        topk = min(int(self.capacity_factor), self.num_routed)
        gating_with_bias = gating_prob + self.router_bias.float().unsqueeze(0)
        _, topk_indices = torch.topk(gating_with_bias, k=topk, dim=-1, sorted=False)
        routed_weights = gating_prob.gather(1, topk_indices)
        if self.router_scoring_func == "sigmoid" and topk > 1:
            routed_weights = routed_weights / (routed_weights.sum(dim=-1, keepdim=True) + epsilon)
        routed_weights = (routed_weights * self.router_scale).to(x_flat.dtype)

        aggregated_output = torch.zeros_like(x_flat)
        aggregated_gate = torch.zeros(num_tokens, dtype=x_flat.dtype, device=x.device)
        for expert_idx, expert in enumerate(self.experts):
            selected_mask = topk_indices == expert_idx
            if not selected_mask.any():
                continue
            n_idx, top = torch.where(selected_mask)
            expert_output = expert(x_flat[n_idx].contiguous())
            contrib = expert_output * routed_weights[n_idx, top].unsqueeze(-1)
            aggregated_output[n_idx] = aggregated_output[n_idx] + contrib
            aggregated_gate[n_idx] = aggregated_gate[n_idx] + routed_weights[n_idx, top]

        return aggregated_output / (aggregated_gate.unsqueeze(-1) + epsilon)


class MoEVisionBlock(nn.Module):
    def __init__(
        self,
        config: SimpleNamespace,
        layer_number: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.attn = DotsVisionAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )
        self.norm_1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.norm_2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        is_moe = (
            config.pyramid_num_routed
            and layer_number < len(config.pyramid_num_routed)
            and config.pyramid_num_routed[layer_number] > 0
        )
        if is_moe:
            self.mlp = MoESwiGLUFFN(
                config,
                layer_number,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = _dots_swiglu(
                config,
                config.intermediate_size,
                quant_config,
                f"{prefix}.mlp",
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm_1(hidden_states),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
        )
        hidden_states = hidden_states + self.mlp(self.norm_2(hidden_states))
        return hidden_states


class DotsMoEVisionTransformer(nn.Module):
    def __init__(
        self,
        config: SimpleNamespace,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.out_hidden_size = config.hidden_size
        self.patch_embed = DotsPatchEmbed(config)
        head_dim = config.embed_dim // config.num_attention_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            [
                MoEVisionBlock(
                    config,
                    layer_idx,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}",
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.post_trunk_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps) if config.post_norm else None
        self.adapter = PatchMerger(
            dim=config.adapter_out_dim,
            context_dim=config.adapter_in_dim,
            spatial_merge_size=config.adapter_merge_size,
            prefix=f"{prefix}.adapter.mlp",
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def get_pos_ids_by_grid(self, grid_thw: list[list[int]]) -> list[torch.Tensor]:
        rope_merge_size = self.spatial_merge_size if self.config.pre_pixel_shuffle else 1
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // rope_merge_size,
                rope_merge_size,
                w // rope_merge_size,
                rope_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3).flatten()
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // rope_merge_size,
                rope_merge_size,
                w // rope_merge_size,
                rope_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3).flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        return pos_ids

    def rot_pos_emb(self, grid_thw: list[list[int]]) -> torch.Tensor:
        pos_ids = torch.cat(self.get_pos_ids_by_grid(grid_thw), dim=0)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        pos_ids = pos_ids.to(rotary_pos_emb_full.device)
        return rotary_pos_emb_full[pos_ids].flatten(1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: list[list[int]],
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        hidden_states = self.patch_embed(pixel_values)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        grid_tensor = torch.tensor(
            grid_thw,
            device=hidden_states.device,
            dtype=torch.long,
        )
        cu_seqlens = torch.repeat_interleave(
            grid_tensor[:, 1] * grid_tensor[:, 2],
            grid_tensor[:, 0],
        ).cumsum(
            dim=0,
            dtype=grid_tensor.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )
        if self.post_trunk_norm is not None:
            hidden_states = self.post_trunk_norm(hidden_states)
        return self.adapter(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = self.state_dict(keep_vars=True)
        expected_checkpoint_names: set[str] = set()
        for name in params_dict:
            if "fc13." in name:
                expected_checkpoint_names.add(name.replace("fc13.", "fc1."))
                expected_checkpoint_names.add(name.replace("fc13.", "fc3."))
            else:
                expected_checkpoint_names.add(name)

        loaded_params: set[str] = set()
        loaded_checkpoint_names: set[str] = set()
        unexpected: set[str] = set()
        for checkpoint_name, loaded_weight in weights:
            name = checkpoint_name
            shard_id: int | None = None
            if "fc1." in name:
                name = name.replace("fc1.", "fc13.")
                shard_id = 0
            elif "fc3." in name:
                name = name.replace("fc3.", "fc13.")
                shard_id = 1

            if (
                name.endswith(".bias")
                and name not in params_dict
                and name.removesuffix("bias") + "weight" in params_dict
            ):
                continue
            if name not in params_dict:
                unexpected.add(checkpoint_name)
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(name)
            loaded_checkpoint_names.add(checkpoint_name)

        missing = expected_checkpoint_names - loaded_checkpoint_names
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={sorted(missing)}")
            if unexpected:
                details.append(f"unexpected={sorted(unexpected)}")
            raise ValueError("Invalid Dots3 Note vision checkpoint: " + "; ".join(details))
        return loaded_params


@MULTIMODAL_REGISTRY.register_processor(
    Dots3NoteMultiModalProcessor,
    info=Dots3NoteProcessingInfo,
    dummy_inputs=Dots3NoteDummyInputsBuilder,
)
class Dots3NoteForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            "vision_encoder.": "visual.",
            "audio_encoder.": "audio_tower.",
        }
    )
    supports_encoder_tp_data = True

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return f"{IMAGE_START}{IMAGE_PAD}{IMAGE_END}"
        if modality.startswith("audio"):
            return f"{AUDIO_START}{AUDIO_PAD}{AUDIO_END}"
        if modality.startswith("video"):
            return VIDEO_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        model_config = vllm_config.model_config
        self.config = model_config.hf_config
        self.quant_config = vllm_config.quant_config
        multimodal_config = model_config.multimodal_config
        assert multimodal_config is not None
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"

        vision_values = load_note_config_section(model_config.model, model_config.revision, "vision_config")
        audio_values = load_note_config_section(model_config.model, model_config.revision, "audio_config")
        video_enabled = vision_values is not None and multimodal_config.get_limit_per_prompt("video") > 0
        image_enabled = vision_values is not None and (
            multimodal_config.get_limit_per_prompt("image") > 0 or video_enabled
        )
        audio_enabled = audio_values is not None and (
            multimodal_config.get_limit_per_prompt("audio") > 0 or video_enabled
        )

        self.visual: DotsMoEVisionTransformer | None = None
        self.audio_tower: Dots3NoteAudioTower | None = None
        with self._mark_tower_model(vllm_config, {"image", "audio", "video"}):
            if image_enabled:
                assert vision_values is not None
                vision_config = SimpleNamespace(**vision_values)
                self.visual = DotsMoEVisionTransformer(
                    vision_config,
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "visual"),
                )
            if audio_enabled:
                assert audio_values is not None
                audio_config = SimpleNamespace(**audio_values)
                self.audio_tower = Dots3NoteAudioTower(audio_config)

        with self._mark_language_model(vllm_config):
            self.language_model = Dots3NoteLanguageModelForCausalLM(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

        disabled_prefixes: dict[str, None] = {}
        if self.visual is None:
            disabled_prefixes["visual."] = None
        if self.audio_tower is None:
            disabled_prefixes["audio_tower."] = None
        if disabled_prefixes:
            self.hf_to_vllm_mapper = self.hf_to_vllm_mapper | WeightsMapper(orig_to_new_prefix=disabled_prefixes)

    def _process_image_input(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if self.visual is None:
            return ()
        pixel_values = pixel_values.to(dtype=self.visual.dtype)
        grid_list = image_grid_thw.tolist()
        if self.use_data_parallel:
            return tuple(
                run_dp_sharded_mrope_vision_model(
                    self.visual,
                    pixel_values,
                    grid_list,
                    rope_type="rope_3d",
                )
            )
        image_embeds = self.visual(pixel_values, grid_list)
        merge_size = self.visual.spatial_merge_size
        sizes = (image_grid_thw.prod(-1) // merge_size**2).tolist()
        return image_embeds.split(sizes)

    def _process_audio_input(
        self,
        audio_values: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if self.audio_tower is None:
            return ()
        if audio_values.dtype != torch.int32:
            raise TypeError(f"NOTE audio values must carry float32 waveform bits as int32, got {audio_values.dtype}")
        waveforms = audio_values.contiguous().view(torch.float32).cpu()
        features = prepare_audio_features(
            waveforms.split(audio_lengths.cpu().tolist()),
            self.audio_tower.config,
        )
        audio_features = features["audio_features"].to(
            device=self.audio_tower.device,
            dtype=self.audio_tower.dtype,
        )
        return self.audio_tower(
            audio_features,
            features["audio_sample_lens"],
            features["audio_segment_counts"],
            features["audio_token_lengths"],
        )

    def _process_video_input(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        audio_values: torch.Tensor,
        audio_lengths: torch.Tensor,
        modalities: torch.Tensor,
        frame_counts: torch.Tensor,
        audio_counts: torch.Tensor,
        emission_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        image_embeds = self._process_image_input(pixel_values, image_grid_thw)
        audio_embeds = self._process_audio_input(audio_values, audio_lengths) if audio_lengths.numel() else ()
        orders = modalities.tolist()
        outputs: list[torch.Tensor] = []
        image_idx = audio_idx = order_idx = 0
        for num_frames, num_audios, num_emissions in zip(
            frame_counts.tolist(),
            audio_counts.tolist(),
            emission_counts.tolist(),
        ):
            video_parts: list[torch.Tensor] = []
            video_image_start = image_idx
            video_audio_start = audio_idx
            for modality in orders[order_idx : order_idx + num_emissions]:
                if modality == 0:
                    video_parts.append(image_embeds[image_idx])
                    image_idx += 1
                elif modality == 1:
                    if audio_idx >= len(audio_embeds):
                        raise ValueError("NOTE video audio tower output is missing")
                    video_parts.append(audio_embeds[audio_idx])
                    audio_idx += 1
                else:
                    raise ValueError(f"Unknown NOTE video modality id: {modality}")
            if image_idx - video_image_start != num_frames:
                raise ValueError("NOTE video frame order/count mismatch")
            if audio_idx - video_audio_start != num_audios:
                raise ValueError("NOTE video audio order/count mismatch")
            outputs.append(torch.cat(video_parts))
            order_idx += num_emissions
        if image_idx != len(image_embeds) or audio_idx != len(audio_embeds):
            raise ValueError("NOTE video encoder outputs were not fully consumed")
        return tuple(outputs)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        multimodal_embeddings: list[torch.Tensor] = []
        handled: set[str] = set()
        for input_key in kwargs:
            if input_key == "pixel_values" and "image" not in handled:
                pixel_values = kwargs.get("pixel_values")
                image_grid_thw = kwargs.get("image_grid_thw")
                if isinstance(pixel_values, torch.Tensor) and isinstance(image_grid_thw, torch.Tensor):
                    multimodal_embeddings.extend(self._process_image_input(pixel_values, image_grid_thw))
                handled.add("image")
            elif input_key == "audio_values" and "audio" not in handled:
                audio_values = kwargs.get("audio_values")
                audio_lengths = kwargs.get("audio_lengths")
                if isinstance(audio_values, torch.Tensor) and isinstance(audio_lengths, torch.Tensor):
                    multimodal_embeddings.extend(self._process_audio_input(audio_values, audio_lengths))
                handled.add("audio")
            elif input_key == "video_pixel_values" and "video" not in handled:
                video_inputs = (
                    kwargs.get("video_pixel_values"),
                    kwargs.get("video_image_grid_thw"),
                    kwargs.get("video_audio_values"),
                    kwargs.get("video_audio_lengths"),
                    kwargs.get("video_modalities"),
                    kwargs.get("video_frame_counts"),
                    kwargs.get("video_audio_counts"),
                    kwargs.get("video_emission_counts"),
                )
                if all(isinstance(value, torch.Tensor) for value in video_inputs):
                    multimodal_embeddings.extend(
                        self._process_video_input(*video_inputs)  # type: ignore[arg-type]
                    )
                handled.add("video")
        return tuple(multimodal_embeddings)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.language_model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights,
            mapper=self.hf_to_vllm_mapper,
        )

    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(
            language_model="language_model",
            tower_model=["visual", "audio_tower"],
        )
