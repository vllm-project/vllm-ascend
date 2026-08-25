# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyAV decoding for the upstream NOTE video planner without CUDA TorchCodec."""

import io

import numpy as np
from PIL import Image


def decode_audio(video_bytes, sample_rate):
    import av  # type: ignore[import-not-found]

    chunks: list[np.ndarray] = []
    with av.open(io.BytesIO(video_bytes)) as container:
        if not container.streams.audio:
            return None, 0.0
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(container.streams.audio[0]):
            chunks.extend(output.to_ndarray().reshape(-1) for output in resampler.resample(frame))
        chunks.extend(output.to_ndarray().reshape(-1) for output in resampler.resample(None))
    if not chunks:
        return None, 0.0
    pcm = (np.concatenate(chunks).clip(-1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm, len(pcm) / sample_rate


def decode_frames(video, video_bytes, visual_budget, seq_length, jpeg_quality):
    import av  # type: ignore[import-not-found]

    with av.open(io.BytesIO(video_bytes)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 25.0)
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None
            else float(container.duration or 0) / av.time_base
        )
        height, width = stream.height, stream.width
        total_frames = int(stream.frames or 0) or max(1, int(round(duration * fps)))
        count, patch_cap = video._solve_degrade(visual_budget, duration, height, width, fps, seq_length)
        source_patches = max(1, round(height / 28)) * max(1, round(width / 28))
        target_h, target_w = video._compute_target_size(
            height, width, 128 * 28**2, min(patch_cap, source_patches) * 28**2
        )
        count = min(count, total_frames)
        indices = np.linspace(0, total_frames - 1, count).round().astype(int)
        frames: list[tuple[float, Image.Image]] = []
        for index in indices:
            timestamp = index / fps
            container.seek(int(timestamp / stream.time_base), stream=stream, backward=True, any_frame=False)
            best_frame = None
            best_distance = float("inf")
            for frame in container.decode(stream):
                frame_time = float(frame.pts * frame.time_base) if frame.pts else 0.0
                distance = abs(frame_time - timestamp)
                if distance < best_distance:
                    best_frame, best_distance = frame, distance
                if frame_time >= timestamp:
                    break
            if best_frame is None:
                raise ValueError(f"Unable to decode Dots3 Note video frame {index}")
            image = best_frame.to_image().convert("RGB")
            if image.size != (target_w, target_h):
                image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
            # NOTE training uses a uniform sampled timeline for audio grouping.
            sampled_timestamp = round(len(frames) / round(count / duration, 4), 3)
            frames.append((sampled_timestamp, video._jpeg_roundtrip(image, jpeg_quality)))
    return frames, duration
