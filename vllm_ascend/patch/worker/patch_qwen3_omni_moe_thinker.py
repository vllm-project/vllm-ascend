from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration
)
from vllm.model_executor.models.utils import WeightsMapper


Qwen3OmniMoeThinkerForConditionalGeneration.hf_to_vllm_mapper = WeightsMapper(
    orig_to_new_prefix={
        "thinker.lm_head.": "language_model.lm_head.",
        "thinker.model.": "language_model.model.",
        "thinker.": "",
        "lm_head.": "language_model.lm_head.",
        "model.": "language_model.model.",
    }
)

Qwen3OmniMoeThinkerForConditionalGeneration.packed_modules_mapping = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

import torch
import torch.nn as nn
import torch_npu
import sys
from torch.nn import functional as F
from typing import Optional
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoePreTrainedModel, SinusoidsPositionEmbedding, _get_feat_extract_output_lengths
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeAudioEncoderConfig
from transformers.utils import auto_docstring

class NPUQwen3OmniMoeAudioAttention(nn.Module):
    def __init__(self, config, quant_config=None, prefix=''):
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        self.dropout = config.attention_dropout
        self.head_dim = self.embed_dim // config.encoder_attention_heads
        self.num_key_value_groups = 1
        self.config = config

        if (self.head_dim * self.num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)


    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        seq_length, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        key_states = self.k_proj(hidden_states).reshape(seq_length, self.num_heads, -1)
        value_states = self.v_proj(hidden_states).reshape(seq_length, self.num_heads, -1)

        q = query_states.reshape(seq_length, self.num_heads, -1)
        k = key_states.reshape(seq_length, self.num_heads, -1)
        v = value_states.reshape(seq_length, self.num_heads, -1)

        attn_output = torch_npu.npu_fusion_attention(
            q, k, v, self.num_heads, 'TND',
            pse=None,
            padding_mask=None,
            atten_mask=None,
            scale=self.scaling,
            pre_tockens=2147483547,
            next_tockens=0,
            keep_prob=1.0,
            inner_precise=0,
            sparse_mode=0,
            actual_seq_qlen=cu_seqlens,
            actual_seq_kvlen=cu_seqlens,
            )[0]

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.out_proj(attn_output)
        return attn_output

def _apply_transformers_audio_attention_patch():
    import torch
    if not hasattr(torch, 'npu') or not torch.npu.is_available():
        print("[vLLM-Ascend] NPU not available, skipping audio attention patch.")
        return

    try:
        import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as modeling_module

        modeling_module.Qwen3OmniMoeAudioAttention = NPUQwen3OmniMoeAudioAttention
        
        print("[vLLM-Ascend] Successfully patched transformers Qwen3OmniMoeAudioAttention with NPU-optimized version.")
    except ImportError:
        print("[vLLM-Ascend] transformers Qwen3OmniMoe module not available, skip patch.")
    except Exception as e:
        print(f"[vLLM-Ascend] Failed to patch transformers audio attention: {e}")

class NPUQwen3OmniMoeAudioEncoder(Qwen3OmniMoePreTrainedModel):
    config: Qwen3OmniMoeAudioEncoderConfig
    main_input_name = "input_features"
    _no_split_modules = ["Qwen3OmniMoeAudioEncoderLayer"]
    _supports_sdpa = True

    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig):
        super().__init__(config)
        self.dropout = config.dropout

        embed_dim = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.n_window = config.n_window
        self.positional_embedding = SinusoidsPositionEmbedding(self.max_source_positions, embed_dim)
        self.layers = nn.ModuleList([Qwen3OmniMoeAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(config.d_model)
        self.gradient_checkpointing = False
        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * ((((config.num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2),
            config.d_model,
            bias=False,
        )
        self.proj1 = nn.Linear(config.d_model, config.d_model)
        self.act = ACT2FN[config.activation_function]
        self.proj2 = nn.Linear(config.d_model, config.output_dim)
        self.n_window_infer = self.config.n_window_infer
        self.conv_chunksize = self.config.conv_chunksize
        # Initialize weights and apply final processing
        self.post_init()

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def get_input_embeddings(self) -> nn.Module:
        return self.conv1

    def set_input_embeddings(self, value: nn.Module):
        self.conv1 = value

    def _prepare_attention_mask(self, inputs_tensor: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        # Flash Attention 2 doesn't need a 4D mask and relies on `cu_seqlens/max_seqlen`
        # NOTE: the created attention masl only approximates the ragged FA2 attention by
        # allowing bidirectional attention within `cu_seqlens` blocks, and not attending between
        # blocks. Though it will not be a 100% match for FA2's `varlen` path
        if self.config._attn_implementation == "flash_attention_2":
            return None

        seq_length = inputs_tensor.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(inputs_tensor.dtype).min,
            device=inputs_tensor.device,
            dtype=inputs_tensor.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
        return attention_mask

    @auto_docstring
    def forward(
        self,
        input_features,
        feature_lens=None,
        aftercnn_lens=None,
    ):
        r"""
        feature_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length
        aftercnn_lens (`torch.LongTensor` of shape `(batch_size,)`):
            mel length after cnn
        """
        aftercnn_lens = _get_feat_extract_output_lengths(feature_lens)
        chunk_num = torch.ceil(feature_lens / (self.n_window * 2)).long()

        chunk_lengths = torch.tensor(
            [self.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
            device=feature_lens.device,
        )
        # print("[debug] --------------------- chunk_lengths", chunk_lengths)


        tail_chunk_index = F.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]

        # print("[debug] tail_chunk_index", tail_chunk_index)
        chunk_lengths[tail_chunk_index] = feature_lens % (self.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = self.n_window * 2

        chunk_list = input_features.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths) # CPU or triton
        # print("[debug] feature_lens_after_cnn", feature_lens_after_cnn)
        padded_mask_after_cnn = torch.ones((feature_lens_after_cnn.shape[0], 13), dtype=torch.bool, device=padded_feature.device)
        for idx in tail_chunk_index:
            padded_mask_after_cnn[idx, feature_lens_after_cnn[idx]:] = False

        # padded_mask_after_cnn_bak = nn.utils.rnn.pad_sequence(
        # padded_mask_after_cnn = nn.utils.rnn.pad_sequence( 
        #     [torch.ones(length, dtype=torch.bool, device=padded_feature.device) for length in feature_lens_after_cnn], # 重构, chunk_lengths是定长？
        #     batch_first=True,
        # )
        # print("[debug] compare", (padded_mask_after_cnn_bak - padded_mask_after_cnn).max(), (padded_mask_after_cnn_bak - padded_mask_after_cnn).min())
        # print("[debug] padded_mask_after_cnn", padded_mask_after_cnn, padded_mask_after_cnn.shape)
        padded_feature = padded_feature.unsqueeze(1)
        # Split to chunk to avoid OOM during convolution
        padded_embeds = []
        for chunk in padded_feature.split(self.conv_chunksize, dim=0):
            padded_embed = F.gelu(self.conv2d1(chunk))
            padded_embed = F.gelu(self.conv2d2(padded_embed))
            padded_embed = F.gelu(self.conv2d3(padded_embed))
            padded_embeds.append(padded_embed)
        padded_embed = torch.cat(padded_embeds, dim=0)
        b, c, f, t = padded_embed.size()
        padded_embed = self.conv_out(padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f))

        positional_embedding = (
            self.positional_embedding.positional_embedding[: padded_embed.shape[1], :]
            .unsqueeze(0)
            .to(padded_embed.dtype)
        )
        padded_embed = padded_embed + positional_embedding
        if padded_mask_after_cnn.shape[1] > padded_embed.shape[1]:
            padded_mask_after_cnn = padded_mask_after_cnn[:, :padded_embed.shape[1]]
        hidden_states = padded_embed[padded_mask_after_cnn]
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (self.n_window_infer // (self.n_window * 2))
        for cnn_len in aftercnn_lens:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]

        cu_seqlens = torch.tensor(cu_chunk_lens, device=aftercnn_lens.device).cumsum(-1, dtype=torch.int32).tolist()

        for encoder_layer in self.layers:
            layer_outputs = encoder_layer(
                hidden_states,
                cu_seqlens,
            )

            hidden_states = layer_outputs[0]

        hidden_states = self.ln_post(hidden_states)
        hidden_states = self.proj1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.proj2(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states)

    def padded_and_mask_function(self, tensor_list, tensor_len, padding_value=0, padding_side="right"):
        """
        Pads a sequence of tensors to their maximum length on indicated `padding_side`.
        Then prepares a mask so that pad tokens are not attended to.
        """
        max_len = tensor_len.max()
        dim = tensor_list[0].shape[0]
        padded_tensor = torch.full(
            size=(len(tensor_list), dim, max_len),
            fill_value=padding_value,
            dtype=self.dtype,
            device=tensor_list[0].device,
        )

        batch_mask = torch.zeros(
            (len(tensor_len), max_len),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(tensor_len):
            batch_mask[i, :length] = 1
            padded_tensor[i, :, :length] = tensor_list[i]

        feature_lens_after_cnn = (tensor_len - 1) // 2 + 1
        max_len_after_cnn = feature_lens_after_cnn.max()
        batch_mask_after_cnn = torch.zeros(
            (len(tensor_len), max_len_after_cnn),
            dtype=torch.long,
            device=padded_tensor.device,
        )
        for i, length in enumerate(feature_lens_after_cnn):
            batch_mask_after_cnn[i, :length] = 1
        return (
            padded_tensor,
            batch_mask.unsqueeze(1),
            batch_mask_after_cnn.bool(),
        )

    # Ignore copy
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1
        return input_lengths, output_lengths

def _apply_transformers_audio_encoder_patch():
    if not hasattr(torch, 'npu') or not torch.npu.is_available():
        print("[vLLM-Ascend] NPU not available, skipping audio encoder patch.")
        return

    try:
        import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as modeling_module

        OriginalEncoder = modeling_module.Qwen3OmniMoeAudioEncoder

        modeling_module.Qwen3OmniMoeAudioEncoder = NPUQwen3OmniMoeAudioEncoder
        
        print(f"[vLLM-Ascend] Successfully patched transformers Qwen3OmniMoeAudioEncoder "
              f"(Original: {id(OriginalEncoder)}, New: {id(NPUQwen3OmniMoeAudioEncoder)})")
    except ImportError:
        print("[vLLM-Ascend] transformers Qwen3OmniMoe module not available, skip audio encoder patch.")
    except Exception as e:
        print(f"[vLLM-Ascend] Failed to patch transformers audio encoder: {e}")

class Qwen3OmniMoeAudioEncoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3OmniMoeAudioEncoderConfig):
        super().__init__()
        self.embed_dim = config.d_model
        # self.self_attn = Qwen3OmniMoeAudioAttention(config)
        self.self_attn = NPUQwen3OmniMoeAudioAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        return outputs

def _apply_transformers_audio_encoder_layer_patch():
    if not hasattr(torch, 'npu') or not torch.npu.is_available():
        print("[vLLM-Ascend] NPU not available, skipping audio encoder layer patch.")
        return

    try:
        import transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe as modeling_module

        OriginalEncoder = modeling_module.Qwen3OmniMoeAudioEncoderLayer

        modeling_module.Qwen3OmniMoeAudioEncoderLayer = Qwen3OmniMoeAudioEncoderLayer

        print(f"[vLLM-Ascend] Successfully patched transformers Qwen3OmniMoeAudioEncoderLayer "
              f"(Original: {id(OriginalEncoder)}, New: {id(NPUQwen3OmniMoeAudioEncoder)})")
    except ImportError:
        print("[vLLM-Ascend] transformers Qwen3OmniMoe module not available, skip audio encoder patch.")
    except Exception as e:
        print(f"[vLLM-Ascend] Failed to patch transformers audio encoder: {e}")

def _apply_vllm_audio_encoder_patch():
    try:
        import vllm.model_executor.models.qwen3_omni_moe_thinker as thinker_module
        print(f"[vLLM-Ascend] vLLM thinker module loaded from: {thinker_module.__file__}", file=sys.stderr)

        if hasattr(thinker_module, 'Qwen3OmniMoeAudioEncoder'):
            original = thinker_module.Qwen3OmniMoeAudioEncoder
            print(f"[vLLM-Ascend] Original Qwen3OmniMoeAudioEncoder ID: {id(original)}", file=sys.stderr)
            thinker_module.Qwen3OmniMoeAudioEncoder = NPUQwen3OmniMoeAudioEncoder
            print(f"[vLLM-Ascend] New Qwen3OmniMoeAudioEncoder ID: {id(NPUQwen3OmniMoeAudioEncoder)}", file=sys.stderr)
            print(f"[vLLM-Ascend] Now thinker_module.Qwen3OmniMoeAudioEncoder ID: {id(thinker_module.Qwen3OmniMoeAudioEncoder)}", file=sys.stderr)
        else:
            print("[vLLM-Ascend] thinker_module has no Qwen3OmniMoeAudioEncoder attribute", file=sys.stderr)
            print("Available in thinker_module:", dir(thinker_module), file=sys.stderr)
    except ImportError as e:
        print(f"[vLLM-Ascend] Failed to import vllm thinker module: {e}", file=sys.stderr)
    except Exception as e:
        print(f"[vLLM-Ascend] Error patching vLLM module: {e}", file=sys.stderr)


_apply_transformers_audio_attention_patch()
_apply_transformers_audio_encoder_layer_patch()
_apply_transformers_audio_encoder_patch()
_apply_vllm_audio_encoder_patch()
