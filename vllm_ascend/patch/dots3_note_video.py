# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import hashlib
import io
import math
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

ALIGN = 28
FPS_MIN_FRAMES = 4
V2_FPS_CAP = 1.0
V2_FPS_MIN = 0.2
V2_PF_FLOOR = 128
V2_PF_CEIL = 1024
V2_FRAME_OVERHEAD = 15
V2_FIXED_OVERHEAD = 64
V2_OVERHEAD = 2240
VIDEO_JPEG_QUALITY = 85
AUDIO_SAMPLES_PER_TOKEN = 1280
INTERLEAVE_SEG_MIN_SEC = 1.0
SYSTEM_BLOCK = "<|system|>You are a helpful assistant.<|endofsystem|>\n"
VIDEO_TRAINING_MARKER = "<video_0>"
ROLE_WRAP_TOKENS = 2


@dataclass(frozen=True)
class Dots3NoteVideoPreprocessResult:
    frames: list[Image.Image]
    timestamps: list[float]
    audio_segments: list[np.ndarray]
    layout: list[tuple[Literal["image", "audio"], int]]


def format_timestamp(seconds: float) -> str:
    total_cs = max(0, int(round(seconds * 100)))
    hours = total_cs // 360000
    minutes = (total_cs // 6000) % 60
    secs = (total_cs // 100) % 60
    centiseconds = total_cs % 100
    return f"<{hours:02d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}>"


def _compute_target_size(
    height: int,
    width: int,
    min_pixels: int,
    max_pixels: int,
) -> tuple[int, int]:
    target_h = max(ALIGN, round(height / ALIGN) * ALIGN)
    target_w = max(ALIGN, round(width / ALIGN) * ALIGN)
    if target_h * target_w > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        target_h = max(ALIGN, math.floor(height / scale / ALIGN) * ALIGN)
        target_w = max(ALIGN, math.floor(width / scale / ALIGN) * ALIGN)
    elif target_h * target_w < min_pixels:
        scale = math.sqrt(min_pixels / max(1, height * width))
        target_h = math.ceil(height * scale / ALIGN) * ALIGN
        target_w = math.ceil(width * scale / ALIGN) * ALIGN
        if target_h * target_w > max_pixels:
            scale = math.sqrt(target_h * target_w / max_pixels)
            target_h = max(ALIGN, math.floor(target_h / scale / ALIGN) * ALIGN)
            target_w = max(ALIGN, math.floor(target_w / scale / ALIGN) * ALIGN)
    return target_h, target_w


def _real_patches_at(height: int, width: int, patch_cap: int) -> int:
    target_h, target_w = _compute_target_size(
        height,
        width,
        V2_PF_FLOOR * ALIGN * ALIGN,
        max(V2_PF_FLOOR, patch_cap) * ALIGN * ALIGN,
    )
    return (target_h // ALIGN) * (target_w // ALIGN)


def _frame_hardcap(seq_length: int) -> int:
    required = max(
        1,
        (seq_length - V2_OVERHEAD) // (V2_PF_FLOOR + V2_FRAME_OVERHEAD),
    )
    hardcap = 1024
    while hardcap < required:
        hardcap <<= 1
    return hardcap


def solve_v2_plan(
    visual_budget: int,
    duration: float,
    height: int,
    width: int,
    source_fps: float,
    *,
    seq_length: int | None = None,
) -> tuple[int, int, int]:
    """Return frame count and train-aligned target height/width."""
    aligned_h = max(ALIGN, round(height / ALIGN) * ALIGN)
    aligned_w = max(ALIGN, round(width / ALIGN) * ALIGN)
    source_patch_cap = (aligned_h // ALIGN) * (aligned_w // ALIGN)
    fps_cap = min(V2_FPS_CAP, max(source_fps, 1e-6))
    patch_cap = min(V2_PF_CEIL, max(source_patch_cap, V2_PF_FLOOR))
    hardcap = _frame_hardcap(seq_length or visual_budget + V2_OVERHEAD)

    def usage(ratio: float) -> tuple[int, float, int]:
        fps = V2_FPS_MIN + ratio * (fps_cap - V2_FPS_MIN)
        patches = V2_PF_FLOOR + ratio * (patch_cap - V2_PF_FLOOR)
        num_frames = max(
            FPS_MIN_FRAMES,
            min(int(round(duration * fps)), hardcap),
        )
        real_patches = _real_patches_at(height, width, int(round(patches)))
        return num_frames * (real_patches + V2_FRAME_OVERHEAD), fps, int(round(patches))

    if usage(1.0)[0] <= visual_budget:
        _, _, target_patches = usage(1.0)
        num_frames = usage(1.0)[0] // (_real_patches_at(height, width, target_patches) + V2_FRAME_OVERHEAD)
    elif usage(0.0)[0] > visual_budget:
        frame_cost = _real_patches_at(height, width, V2_PF_FLOOR) + V2_FRAME_OVERHEAD
        num_frames = max(
            FPS_MIN_FRAMES,
            min(visual_budget // frame_cost, hardcap),
        )
        target_patches = V2_PF_FLOOR
    else:
        low, high = 0.0, 1.0
        for _ in range(50):
            middle = (low + high) / 2
            if usage(middle)[0] <= visual_budget:
                low = middle
            else:
                high = middle
        _, _, target_patches = usage(low)
        frame_cost = _real_patches_at(height, width, target_patches) + V2_FRAME_OVERHEAD
        num_frames = max(
            FPS_MIN_FRAMES,
            usage(low)[0] // frame_cost,
        )

    max_pixels = min(
        target_patches * ALIGN * ALIGN,
        source_patch_cap * ALIGN * ALIGN,
    )
    target_h, target_w = _compute_target_size(
        height,
        width,
        V2_PF_FLOOR * ALIGN * ALIGN,
        max_pixels,
    )
    return num_frames, target_h, target_w


def _video_metadata(video_bytes: bytes) -> tuple[float, int, int, float, int, bool]:
    import av  # type: ignore[import-not-found]

    with av.open(io.BytesIO(video_bytes)) as container:
        if not container.streams.video:
            raise ValueError("Dots3 Note video input has no video stream")
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 0) or 25.0
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None
            else float(container.duration or 0) / av.time_base
        )
        total_frames = int(stream.frames or 0) or max(1, int(round(duration * fps)))
        height = int(stream.height)
        width = int(stream.width)
        has_audio = bool(container.streams.audio)
    if duration <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"Invalid Dots3 Note video metadata: duration={duration}, height={height}, width={width}")
    return duration, height, width, fps, total_frames, has_audio


def _uniform_indices(total_frames: int, num_frames: int) -> list[int]:
    num_frames = max(1, min(num_frames, total_frames))
    if num_frames == 1:
        return [0]
    step = (total_frames - 1) / (num_frames - 1)
    return sorted({int(round(index * step)) for index in range(num_frames)})


def _decode_frames(
    video_bytes: bytes,
    indices: list[int],
    fps: float,
    target_h: int,
    target_w: int,
) -> list[Image.Image]:
    import av

    frames: list[Image.Image] = []
    with av.open(io.BytesIO(video_bytes)) as container:
        stream = container.streams.video[0]
        time_base = float(stream.time_base)
        for index in indices:
            target_time = index / fps
            target_pts = max(0, int(target_time / time_base))
            container.seek(target_pts, stream=stream, backward=True, any_frame=False)
            stream.codec_context.flush_buffers()
            best_frame = None
            best_distance = float("inf")
            for frame in container.decode(stream):
                frame_time = frame.time
                if frame_time is None:
                    frame_time = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
                distance = abs(frame_time - target_time)
                if distance < best_distance:
                    best_frame = frame
                    best_distance = distance
                if frame_time >= target_time:
                    break
            if best_frame is None:
                raise ValueError(f"Unable to decode Dots3 Note video frame {index}")
            image = best_frame.to_image().convert("RGB")
            if image.size != (target_w, target_h):
                image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
            jpeg = io.BytesIO()
            image.save(jpeg, format="JPEG", quality=VIDEO_JPEG_QUALITY)
            jpeg.seek(0)
            with Image.open(jpeg) as decoded:
                frames.append(decoded.convert("RGB"))
    return frames


def _decode_audio(video_bytes: bytes, sample_rate: int) -> np.ndarray | None:
    import av

    chunks: list[np.ndarray] = []
    with av.open(io.BytesIO(video_bytes)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(stream):
            for output in resampler.resample(frame):
                chunks.append(output.to_ndarray().reshape(-1))
        for output in resampler.resample(None):
            chunks.append(output.to_ndarray().reshape(-1))
    if not chunks:
        return None
    waveform = np.concatenate(chunks).astype(np.float32)
    pcm = (waveform.clip(-1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm.astype(np.float32) / 32768.0


def _audio_tokens(num_samples: int) -> int:
    return 0 if num_samples <= 0 else math.ceil(num_samples / AUDIO_SAMPLES_PER_TOKEN) + 2


def _reference_prompt_overhead(tokenizer) -> int:
    system_tokens = len(tokenizer.encode(SYSTEM_BLOCK, add_special_tokens=False))
    marker_tokens = len(tokenizer.encode(VIDEO_TRAINING_MARKER, add_special_tokens=False))
    return system_tokens + ROLE_WRAP_TOKENS + marker_tokens + V2_FIXED_OVERHEAD


def _flatten_seed(video_bytes: bytes, question: str) -> int:
    video_id = hashlib.sha1(video_bytes).hexdigest()
    record_key = hashlib.sha1(f"{video_id}|{question}".encode()).hexdigest()
    digest = hashlib.sha1(f"42|flatten|{record_key}".encode()).hexdigest()
    return int(digest[:8], 16)


def _group_bounds(
    num_frames: int,
    duration: float,
    mode: str,
    rng: random.Random,
) -> list[int]:
    if mode not in {"logk", "eval30", "eval_ek", "whole"}:
        raise ValueError(f"Unsupported Dots3 Note video k_mode: {mode!r}")
    k_max = min(
        num_frames,
        max(1, int(duration // INTERLEAVE_SEG_MIN_SEC)),
    )
    if mode == "whole" or k_max <= 1:
        groups = 1
    elif mode == "eval30":
        groups = int(round(math.sqrt(k_max)))
    elif mode == "eval_ek":
        groups = int(round((k_max - 1) / math.log(k_max)))
    else:
        groups = int(round(math.exp(rng.uniform(0.0, math.log(k_max)))))
    groups = max(1, min(k_max, groups))
    if groups == 1:
        return [0, num_frames]
    if mode == "logk":
        cuts = sorted(rng.sample(range(1, num_frames), groups - 1))
    else:
        cuts = sorted({round(group * num_frames / groups) for group in range(1, groups)})
    return [0, *[cut for cut in cuts if 0 < cut < num_frames], num_frames]


def _build_layout(
    num_frames: int,
    timestamps: list[float],
    waveform: np.ndarray | None,
    sample_rate: int,
    mode: str,
    rng: random.Random,
) -> tuple[list[np.ndarray], list[tuple[Literal["image", "audio"], int]]]:
    if waveform is None:
        return [], [("image", index) for index in range(num_frames)]

    audio_duration = len(waveform) / sample_rate
    bounds = _group_bounds(num_frames, audio_duration, mode, rng)
    segments: list[np.ndarray] = []
    layout: list[tuple[Literal["image", "audio"], int]] = []
    for group in range(len(bounds) - 1):
        frame_start, frame_end = bounds[group], bounds[group + 1]
        layout.extend(("image", index) for index in range(frame_start, frame_end))
        start_time = 0.0 if group == 0 else timestamps[frame_start]
        end_time = audio_duration if group == len(bounds) - 2 else timestamps[bounds[group + 1]]
        sample_start = max(0, int(round(start_time * sample_rate)))
        sample_end = min(len(waveform), int(round(end_time * sample_rate)))
        if sample_end > sample_start:
            segments.append(waveform[sample_start:sample_end].copy())
            layout.append(("audio", len(segments) - 1))
    return segments, layout


def preprocess_video(
    video: object,
    *,
    prompt: str,
    question: str,
    tokenizer,
    seq: int = 131072,
    output_reserve: int | None = None,
    max_new_tokens: int = 0,
    audio_cap: float = 1.0,
    audio_sr: int = 16000,
    k_mode: str = "eval_ek",
) -> Dots3NoteVideoPreprocessResult:
    """Convert a video into budgeted frame, audio, and prompt features."""
    if (
        seq <= 0
        or max_new_tokens < 0
        or (output_reserve is not None and output_reserve < 0)
        or audio_cap < 0
        or audio_sr <= 0
    ):
        raise ValueError("Invalid Dots3 Note video preprocessing limits")
    if k_mode not in {"logk", "eval30", "eval_ek", "whole"}:
        raise ValueError(f"Unsupported Dots3 Note video k_mode: {k_mode!r}")
    reserve = max(seq // 4 if output_reserve is None else output_reserve, max_new_tokens)
    if reserve >= seq:
        raise ValueError("Dots3 Note video output reserve must be smaller than seq")

    frames_data, metadata = video if isinstance(video, tuple) else (video, {})
    metadata = metadata or {}
    raw_bytes = metadata.get("original_video_bytes")
    if not isinstance(raw_bytes, bytes):
        arrays = list(frames_data)
        if not arrays:
            raise ValueError("Dots3 Note video input has no frames")
        fps = float(metadata.get("fps", 1.0) or 1.0)
        default_duration = len(arrays) / fps
        duration = float(metadata.get("duration", default_duration) or default_duration)
        indices = _uniform_indices(len(arrays), min(len(arrays), FPS_MIN_FRAMES))
        frames = [Image.fromarray(np.asarray(arrays[index])).convert("RGB") for index in indices]
        timestamps = [round(index / fps, 3) for index in indices]
        return Dots3NoteVideoPreprocessResult(
            frames=frames,
            timestamps=timestamps,
            audio_segments=[],
            layout=[("image", index) for index in range(len(frames))],
        )

    duration, height, width, fps, total_frames, has_audio = _video_metadata(raw_bytes)
    rng = random.Random(_flatten_seed(raw_bytes, question))
    available = seq - reserve
    prompt_overhead = _reference_prompt_overhead(tokenizer)
    waveform = None
    if has_audio and audio_cap > 0:
        try:
            waveform = _decode_audio(raw_bytes, audio_sr)
        except Exception:
            waveform = None

    audio_duration = len(waveform) / audio_sr if waveform is not None else 0.0
    pure_audio_tokens = _audio_tokens(len(waveform)) if waveform is not None else 0
    frame_upper_bound = max(1, int(duration * V2_FPS_CAP))
    k_upper_bound = min(
        frame_upper_bound,
        max(1, int(audio_duration // INTERLEAVE_SEG_MIN_SEC)),
    )
    audio_budget = pure_audio_tokens + 3 * k_upper_bound
    minimum_visual = FPS_MIN_FRAMES * (V2_PF_FLOOR + V2_FRAME_OVERHEAD)
    use_audio = (
        waveform is not None
        and pure_audio_tokens <= audio_cap * available
        and audio_budget + minimum_visual + V2_OVERHEAD <= available
    )
    if not use_audio:
        waveform = None
        audio_budget = 0

    visual_budget = max(
        V2_PF_FLOOR + V2_FRAME_OVERHEAD,
        available - prompt_overhead - audio_budget,
    )
    num_frames, target_h, target_w = solve_v2_plan(
        visual_budget,
        duration,
        height,
        width,
        fps,
        seq_length=available,
    )
    indices = _uniform_indices(total_frames, num_frames)
    frames = _decode_frames(raw_bytes, indices, fps, target_h, target_w)
    sampled_fps = len(frames) / duration
    timestamps = [round(index / sampled_fps, 3) for index in range(len(frames))]
    audio_segments, layout = _build_layout(
        len(frames),
        timestamps,
        waveform,
        audio_sr,
        k_mode,
        rng,
    )
    return Dots3NoteVideoPreprocessResult(
        frames=frames,
        timestamps=timestamps,
        audio_segments=audio_segments,
        layout=layout,
    )
