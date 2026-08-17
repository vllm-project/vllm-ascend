# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.audio_utils import mel_filter_bank
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_ascend.patch.dots3_note_config import Dots3NoteAudioConfig

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160


def compute_audio_token_length(num_samples: int, config: Dots3NoteAudioConfig) -> int:
    stride = HOP_LENGTH * 8 * config.merge_factor
    chunk_samples = config.chunk_seconds * config.sampling_rate
    return sum(
        math.ceil(min(chunk_samples, num_samples - start) / stride) for start in range(0, num_samples, chunk_samples)
    )


@lru_cache(maxsize=1)
def _mel_filters() -> torch.Tensor:
    filters = mel_filter_bank(
        num_frequency_bins=1 + N_FFT // 2,
        num_mel_filters=128,
        min_frequency=0.0,
        max_frequency=SAMPLE_RATE / 2.0,
        sampling_rate=SAMPLE_RATE,
        norm="slaney",
        mel_scale="slaney",
    )
    return torch.from_numpy(filters).T.contiguous().float()


@lru_cache(maxsize=1)
def _hann_window() -> torch.Tensor:
    return torch.hann_window(N_FFT)


def _log_mel_spectrogram(audio: torch.Tensor) -> torch.Tensor:
    stft = torch.stft(
        audio,
        N_FFT,
        HOP_LENGTH,
        window=_hann_window(),
        return_complex=True,
    )
    magnitudes = stft[..., :-1].abs().square()
    mel_spec = _mel_filters() @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def prepare_audio_features(
    audios: Sequence[object],
    config: Dots3NoteAudioConfig,
) -> dict[str, torch.Tensor]:
    if config.sampling_rate != SAMPLE_RATE:
        raise ValueError(f"Dots3 Note audio encoder requires {SAMPLE_RATE} Hz, got {config.sampling_rate}")

    chunk_samples = config.chunk_seconds * config.sampling_rate
    chunk_frames = config.chunk_seconds * 100
    input_features: list[torch.Tensor] = []
    sample_lengths: list[int] = []
    segment_counts: list[int] = []
    token_lengths: list[int] = []

    for audio in audios:
        waveform = torch.as_tensor(audio, dtype=torch.float32).squeeze()
        if waveform.ndim != 1 or waveform.numel() == 0:
            raise ValueError("Dots3 Note audio inputs must be non-empty mono waveforms")

        segment_count = 0
        for start in range(0, waveform.numel(), chunk_samples):
            segment = waveform[start : start + chunk_samples]
            sample_length = int(segment.numel())
            padded = F.pad(segment, (0, chunk_samples - sample_length))
            mel = _log_mel_spectrogram(padded)
            if mel.shape[-1] != chunk_frames:
                raise ValueError(f"Unexpected Dots3 Note mel length {mel.shape[-1]}, expected {chunk_frames}")
            input_features.append(mel)
            sample_lengths.append(sample_length)
            segment_count += 1

        segment_counts.append(segment_count)
        token_lengths.append(compute_audio_token_length(waveform.numel(), config))

    return {
        "audio_features": torch.stack(input_features),
        "audio_sample_lens": torch.tensor(sample_lengths, dtype=torch.long),
        "audio_segment_counts": torch.tensor(segment_counts, dtype=torch.long),
        "audio_token_lengths": torch.tensor(token_lengths, dtype=torch.long),
    }


class Dots3NoteAudioRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class Dots3NoteAudioRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        rope_parameters: dict,
        base_seq_len: int = 0,
    ) -> None:
        super().__init__()
        partial_factor = float(rope_parameters.get("partial_rotary_factor", 1.0))
        rotary_dim = int(head_dim * partial_factor)
        self.rotary_dim = rotary_dim // 2 * 2
        self.rope_theta = float(rope_parameters.get("rope_theta", 10000.0))
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.base_seq_len = max(0, int(base_seq_len))
        self._cache: tuple[int, torch.dtype, torch.device, torch.Tensor, torch.Tensor] | None = None

    @torch.no_grad()
    def forward(
        self,
        seq_len: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cache is not None:
            cached_len, cached_dtype, cached_device, cosine, sine = self._cache
            if cached_len >= seq_len and cached_dtype == dtype and cached_device == device:
                return cosine[:seq_len], sine[:seq_len]

        cache_len = max(seq_len, self.base_seq_len)
        positions = torch.arange(cache_len, device=device)[None, :]
        if self.inv_freq.dtype != torch.float32:
            inv_freq = 1.0 / (
                self.rope_theta
                ** (
                    torch.arange(
                        0,
                        self.rotary_dim,
                        2,
                        dtype=torch.float32,
                        device=device,
                    )
                    / max(self.rotary_dim, 1)
                )
            )
        else:
            inv_freq = self.inv_freq.to(device=device)
        inv_freq = inv_freq[None, :, None]
        device_type = device.type if isinstance(device, torch.device) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            frequencies = (inv_freq.float() @ positions[:, None, :].float()).transpose(1, 2)
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            cosine = embedding.cos().to(dtype)[0]
            sine = embedding.sin().to(dtype)[0]
        self._cache = (cache_len, dtype, device, cosine, sine)
        return cosine[:seq_len], sine[:seq_len]

    def apply(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cosine = cosine[None, :, None, :]
        sine = sine[None, :, None, :]
        query_rot, query_pass = (
            query[..., : self.rotary_dim],
            query[..., self.rotary_dim :],
        )
        key_rot, key_pass = (
            key[..., : self.rotary_dim],
            key[..., self.rotary_dim :],
        )
        query_rot = query_rot * cosine + _rotate_half(query_rot) * sine
        key_rot = key_rot * cosine + _rotate_half(key_rot) * sine
        return torch.cat((query_rot, query_pass), dim=-1), torch.cat((key_rot, key_pass), dim=-1)


@CustomOp.register("dots3_note_audio_attention")
class Dots3NoteAudioAttentionBackend(CustomOp):
    def __init__(self, num_heads: int, head_size: int, scale: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_heads
        self.head_size = head_size
        self.scale = scale

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del cu_seqlens
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        return output.transpose(1, 2).contiguous()

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_native(query, key, value, cu_seqlens)


class Dots3NoteAudioAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rope_parameters: dict,
        base_seq_len: int,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.rotary_embedding = Dots3NoteAudioRotaryEmbedding(
            self.head_dim,
            rope_parameters,
            base_seq_len,
        )
        self.attn = Dots3NoteAudioAttentionBackend(
            num_heads=num_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        query = self.q_proj(hidden_states).view(1, seq_len, self.num_heads, self.head_dim)
        key = self.k_proj(hidden_states).view(1, seq_len, self.num_heads, self.head_dim)
        value = self.v_proj(hidden_states).view(1, seq_len, self.num_heads, self.head_dim)
        cosine, sine = self.rotary_embedding(seq_len, dtype=query.dtype, device=query.device)
        query, key = self.rotary_embedding.apply(query, key, cosine, sine)
        cu_seqlens = torch.tensor([0, seq_len], device=hidden_states.device, dtype=torch.int32)
        output = self.attn(
            query=query,
            key=key,
            value=value,
            cu_seqlens=cu_seqlens,
        )
        return self.out_proj(output.reshape(seq_len, -1))


class Dots3NoteAudioEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_size: int,
        rope_parameters: dict,
        base_seq_len: int,
        *,
        prefix: str,
    ) -> None:
        super().__init__()
        self.self_attn = Dots3NoteAudioAttention(
            hidden_size,
            num_heads,
            rope_parameters,
            base_seq_len,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = Dots3NoteAudioRMSNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, ffn_size * 2)
        self.fc2 = nn.Linear(ffn_size, hidden_size)
        self.final_layer_norm = Dots3NoteAudioRMSNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        gate, up = self.fc1(hidden_states).chunk(2, dim=-1)
        hidden_states = F.silu(gate) * up
        return residual + self.fc2(hidden_states)


class Dots3NoteAudioSpeechEncoder(nn.Module):
    def __init__(self, config: Dots3NoteAudioConfig, *, prefix: str) -> None:
        super().__init__()
        whisper_config = config.whisper_config
        hidden_size = int(whisper_config["d_model"])
        num_heads = int(whisper_config["encoder_attention_heads"])
        ffn_size = int(whisper_config["encoder_ffn_dim"])
        num_layers = int(whisper_config["encoder_layers"])
        num_mel_bins = int(whisper_config["num_mel_bins"])
        if not config.use_conv2d_stem or not config.use_rope:
            raise ValueError("Dots3 Note AE requires the Conv2D stem and RoPE")
        if config.use_causal or not config.use_rms_norm:
            raise ValueError("Dots3 Note AE requires non-causal attention with RMSNorm")
        if whisper_config.get("activation_function") != "swiglu":
            raise ValueError("Dots3 Note AE requires SwiGLU feed-forward layers")

        downsample_size = config.downsample_hidden_size
        self.conv2d1 = nn.Conv2d(1, downsample_size, 3, stride=2, padding=1)
        self.conv2d2 = nn.Conv2d(downsample_size, downsample_size, 3, stride=2, padding=1)
        self.conv2d3 = nn.Conv2d(downsample_size, downsample_size, 3, stride=2, padding=1)
        freq_after = num_mel_bins
        for _ in range(3):
            freq_after = (freq_after + 1) // 2
        self.conv_out = nn.Linear(downsample_size * freq_after, hidden_size, bias=False)
        self.layers = nn.ModuleList(
            Dots3NoteAudioEncoderLayer(
                hidden_size,
                num_heads,
                ffn_size,
                config.rope_parameters,
                int(whisper_config.get("max_source_positions", 0)),
                prefix=f"{prefix}.layers.{layer_idx}",
            )
            for layer_idx in range(num_layers)
        )
        self.layer_norm = Dots3NoteAudioRMSNorm(hidden_size)

    @staticmethod
    def _temporal_mask(hidden_states: torch.Tensor, valid_lengths: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(hidden_states.shape[-1], device=hidden_states.device)
        mask = positions[None, :] < valid_lengths[:, None]
        return hidden_states * mask[:, None, None, :]

    def _conv_stem(self, input_features: torch.Tensor, sample_length: int) -> torch.Tensor:
        hidden_states = input_features.unsqueeze(1)
        valid_length = torch.tensor([sample_length // HOP_LENGTH], device=hidden_states.device)
        hidden_states = self._temporal_mask(hidden_states, valid_length)
        hidden_states = F.gelu(self.conv2d1(hidden_states))
        valid_length = (valid_length + 1) // 2
        hidden_states = self._temporal_mask(hidden_states, valid_length)
        hidden_states = F.gelu(self.conv2d2(hidden_states))
        valid_length = (valid_length + 1) // 2
        hidden_states = self._temporal_mask(hidden_states, valid_length)
        hidden_states = F.gelu(self.conv2d3(hidden_states))
        valid_length = (valid_length + 1) // 2
        hidden_states = self._temporal_mask(hidden_states, valid_length)
        batch_size, channels, frequency, frames = hidden_states.shape
        hidden_states = hidden_states.permute(0, 3, 1, 2).reshape(batch_size, frames, channels * frequency)
        return self.conv_out(hidden_states)

    def forward(self, input_features: torch.Tensor, sample_length: int) -> torch.Tensor:
        hidden_states = self._conv_stem(input_features, sample_length)
        token_length = math.ceil(sample_length / (HOP_LENGTH * 8))
        hidden_states = hidden_states[0, :token_length]
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.layer_norm(hidden_states)


class Dots3NoteDotsAudioEncoder(nn.Module):
    def __init__(self, config: Dots3NoteAudioConfig, *, prefix: str) -> None:
        super().__init__()
        self.speech_encoder = Dots3NoteAudioSpeechEncoder(config, prefix=f"{prefix}.speech_encoder")


class Dots3NoteAudioAdapter(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, output_size),
            nn.GELU(),
            nn.Linear(output_size, output_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)


class Dots3NoteAudioTower(nn.Module):
    def __init__(self, config: Dots3NoteAudioConfig, *, prefix: str = "") -> None:
        super().__init__()
        if config.encoder_type != "dots":
            raise ValueError(f"Unsupported Dots3 Note audio encoder: {config.encoder_type}")
        self.config = config
        self.dots_encoder = Dots3NoteDotsAudioEncoder(config, prefix=f"{prefix}.dots_encoder")
        self.audio_adapter = Dots3NoteAudioAdapter(
            config.whisper_adapter_in_dim,
            config.whisper_adapter_out_dim,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.dots_encoder.speech_encoder.conv2d1.weight.dtype

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_sample_lens: torch.Tensor,
        audio_segment_counts: torch.Tensor,
        audio_token_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        segment_counts = audio_segment_counts.tolist()
        sample_lengths = audio_sample_lens.tolist()
        expected_token_lengths = audio_token_lengths.tolist()
        outputs: list[torch.Tensor] = []
        segment_idx = 0

        for segment_count, expected_token_length in zip(segment_counts, expected_token_lengths):
            chunks: list[torch.Tensor] = []
            for _ in range(segment_count):
                chunks.append(
                    self.dots_encoder.speech_encoder(
                        audio_features[segment_idx : segment_idx + 1].to(self.dtype),
                        int(sample_lengths[segment_idx]),
                    )
                )
                segment_idx += 1
            output = self.audio_adapter(torch.cat(chunks, dim=0))
            if output.shape[0] != expected_token_length:
                raise ValueError(
                    f"Dots3 Note AE output length mismatch: expected {expected_token_length}, got {output.shape[0]}"
                )
            outputs.append(output)

        return tuple(outputs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        expected = set(params)
        loaded: set[str] = set()
        unexpected: set[str] = set()

        for name, loaded_weight in weights:
            param = params.get(name)
            if param is None:
                unexpected.add(name)
                continue
            default_weight_loader(param, loaded_weight)
            loaded.add(name)

        missing = expected - loaded
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing={sorted(missing)}")
            if unexpected:
                details.append(f"unexpected={sorted(unexpected)}")
            raise ValueError("Invalid Dots3 Note audio checkpoint: " + "; ".join(details))
        return loaded
