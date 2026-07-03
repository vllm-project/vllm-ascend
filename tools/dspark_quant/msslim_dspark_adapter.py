# DeepSeek-V4-Flash-DSpark quant adapter for msModelSlim (path A).
# Subclasses DeepSeekV4ModelAdapter; overrides the 4 MTP-1-specific spots with the
# DSpark draft structure (main_proj/main_norm + 3-stage stack + markov/confidence).
# Place at: msmodelslim/model/deepseek_v4/dspark_adapter.py  (+ register via loader)
#
# UNTESTED until full ckpt downloads + a free card. Logic written against the read
# framework; the capture-target-layers calib forward (generate_model_forward) is the
# part most likely to need on-NPU tweaking.
from typing import Any, Generator, Tuple, Dict, Optional
from unittest.mock import patch

import torch
from torch import nn

from msmodelslim.core.base.protocol import ProcessRequest
from msmodelslim.utils.exception import InvalidModelError
from msmodelslim.utils.logging import get_logger
from .convert_fp8_to_bf16 import auto_dequant_state_dict
from .model_adapter import DeepSeekV4ModelAdapter
from .model import Transformer
from ..common.weight_helper import get_state_dict
from ..common.transformers import TransformersForwardBreak
from .dspark_model import DSparkBlock


class DeepSeekV4DSparkModelAdapter(DeepSeekV4ModelAdapter):

    def get_model_pedigree(self) -> str:
        return 'deepseek_v4'

    # ---- ⑤ config: add dspark_* fields + force n_mtp_layers=3 ----
    def _load_config(self, trust_remote_code=False):
        from msmodelslim.utils.security import json_safe_load
        import os.path
        args = super()._load_config(trust_remote_code)
        cfg = json_safe_load(os.path.join(self.model_path, "config.json"))
        args.dspark_block_size = cfg["dspark_block_size"]
        args.dspark_noise_token_id = cfg["dspark_noise_token_id"]
        args.dspark_target_layer_ids = list(cfg["dspark_target_layer_ids"])
        args.dspark_markov_rank = cfg["dspark_markov_rank"]
        # DSpark draft = 3 stages (config n_mtp_layers, else count mtp.* in weight_map)
        args.n_mtp_layers = cfg.get("n_mtp_layers", 3)
        return args

    # ---- ③ scaffold: build DSparkBlock (not Block+MTPLayer) for mtp stages ----
    def load_mtp_decoder_if_not_exist(self, model: nn.Module, layer_prefix: str, mtp_idx: int):
        try:
            return model.mtp[mtp_idx]
        except (IndexError, AttributeError):
            with patch.object(nn.Linear, 'reset_parameters', lambda _self: None):
                get_logger().info('Creating DSpark MTP stage %s', mtp_idx)
                layer_id = self.config.num_hidden_layers + mtp_idx
                blk = DSparkBlock(layer_id, self.config)
                state_dict = get_state_dict(self.model_path, blk, prefix=layer_prefix)
                auto_dequant_state_dict(layer_prefix, state_dict, str(self.model_path))
                blk.load_state_dict(state_dict, strict=False)
                object.__setattr__(blk, "embed", model.embed)
                object.__setattr__(blk, "head", model.head)
                blk.eval()
                model.mtp.append(blk)
            return blk

    # ---- ② calib forward: capture target layers [40,41,42] -> main_x -> DSpark stack ----
    def generate_model_forward(self, model: nn.Module, inputs: Any) -> Generator[ProcessRequest, Any, None]:
        first_block_input: Optional[Tuple] = None

        def break_hook(module, hook_args, hook_kwargs):
            nonlocal first_block_input
            first_block_input = (hook_args, hook_kwargs)
            raise TransformersForwardBreak()

        remove_handler = model.layers[0].register_forward_pre_hook(break_hook, with_kwargs=True, prepend=True)
        try:
            if isinstance(inputs, (list, tuple)):
                model(inputs[0])
            elif isinstance(inputs, dict):
                model(**inputs)
            else:
                model(inputs)
        except TransformersForwardBreak:
            pass
        finally:
            remove_handler.remove()
        if first_block_input is None:
            raise InvalidModelError("Can't get first block input.", action="check model/input")

        args, kwargs = first_block_input
        h, start_pos, input_ids = args
        target_ids = list(self.config.dspark_target_layer_ids)
        captured: Dict[int, torch.Tensor] = {}
        main_x = None
        draft_h = None

        for name, block in self.generate_decoder_layer(model):
            if name.startswith('mtp.'):
                mtp_idx = int(name.split('.')[1])
                if mtp_idx == 0:
                    # main_hidden = concat over target layers of mean-over-hc hidden
                    main_hidden = torch.cat(
                        [captured[i].mean(dim=2) for i in target_ids], dim=-1)
                    draft_h, main_x = block.forward_embed(main_hidden, input_ids)
                draft_args = (draft_h, start_pos, input_ids, main_x)
                draft_h = yield ProcessRequest(name, block, draft_args, kwargs)
                continue
            # target layer
            h = yield ProcessRequest(name, block, args, kwargs)
            layer_idx = int(name.split('.')[1])
            if layer_idx in target_ids:
                captured[layer_idx] = h
            args = (h, start_pos, input_ids)

    # ---- ④ maps: post-process super() to drop MTP-1-only entries + add DSpark ones ----
    def _dspark_fix_mtp_keys(self, mapping: dict):
        """Remove MTP-1-only modules (enorm/hnorm/e_proj/h_proj) from a name->targets
        map and inject DSpark main_norm->main_proj (stage0). norm->head kept only on the
        last stage (DSpark mtp shares one head)."""
        n_mtp = self.config.n_mtp_layers
        for i in range(n_mtp):
            for k in (f"mtp.{i}.enorm", f"mtp.{i}.hnorm"):
                mapping.pop(k, None)
            if i != n_mtp - 1:
                mapping.pop(f"mtp.{i}.norm", None)
        mapping[f"mtp.0.main_norm"] = ["mtp.0.main_proj"]
        return mapping

    def get_ln_fuse_map(self):
        pre, ln = super().get_ln_fuse_map()
        self._dspark_fix_mtp_keys(ln)
        return pre, ln

    def get_rotate_map(self, block_size):
        pre_runs, rot_pairs = super().get_rotate_map(block_size)
        # drop rotations on non-existent MTP-1 modules; keep attn/ffn/hc rotations.
        n_mtp = self.config.n_mtp_layers
        dead = []
        for i in range(n_mtp):
            dead += [f"mtp.{i}.h_proj", f"mtp.{i}.e_proj", f"mtp.{i}.emb.tok_emb"]
            if i != n_mtp - 1:
                dead.append(f"mtp.{i}.head")
        for rp in rot_pairs:
            for d in dead:
                rp.left_rot.pop(d, None)
                rp.right_rot.pop(d, None)
        return pre_runs, rot_pairs

    def get_adapter_config_for_subgraph(self):
        cfgs = super().get_adapter_config_for_subgraph()
        # The base only emits attn/ffn subgraphs (up-down, linear-linear, norm-linear)
        # which DSpark mtp shares structurally -> nothing MTP-1-specific to strip here.
        # main_proj has no smoothquant partner; it is quantized as a plain linear via
        # the yaml `*main_proj*` include (handled in the recipe, not subgraph).
        return cfgs
