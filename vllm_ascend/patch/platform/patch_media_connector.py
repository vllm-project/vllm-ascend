#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Backport vLLM's original-image-mode support to the v0.26 release.

vLLM PR #49159 made ``image_mode`` in ``media_io_kwargs`` override the
connector default and allowed ``None`` to preserve an image's original mode.
Kimi K3's upstream renderer relies on that contract, but vLLM v0.26.0 was cut
before the change. Keep the release behavior aligned with main until the
supported release contains the upstream fix.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import vllm.envs as envs
from PIL import UnidentifiedImageError
from vllm.multimodal.media.base import MediaWithBytes
from vllm.multimodal.media.connector import MediaConnector
from vllm.multimodal.media.image import ImageMediaIO
from vllm.multimodal.media.video import VideoMediaIO
from vllm.multimodal.video import get_video_loader_backend_for_processor

from vllm_ascend.utils import vllm_version_is

_ORIGINAL_CONNECTOR_METHODS_ATTR = "_ascend_original_image_mode_connector_methods"
_ORIGINAL_CONVERT_IMAGE_MODE_ATTR = "_ascend_original_convert_image_mode"


def _make_image_io(
    connector: MediaConnector,
    image_mode: str | None,
) -> ImageMediaIO:
    """Let media_io_kwargs override the call-site image mode."""

    return ImageMediaIO(**({"image_mode": image_mode} | connector.media_io_kwargs.get("image", {})))


@wraps(ImageMediaIO._convert_image_mode)
def _convert_image_mode(
    self: ImageMediaIO,
    image: Any,
) -> Any:
    if isinstance(image, MediaWithBytes):
        image = image.media
    if self.image_mode is None:
        return image

    original = getattr(ImageMediaIO, _ORIGINAL_CONVERT_IMAGE_MODE_ATTR)
    return original(self, image)


@wraps(MediaConnector.fetch_image)
def _fetch_image(
    self: MediaConnector,
    image_url: str,
    *,
    image_mode: str | None = "RGB",
) -> Any:
    image_io = _make_image_io(self, image_mode)
    try:
        return self.load_from_url(
            image_url,
            image_io,
            fetch_timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
        )
    except UnidentifiedImageError as exc:
        raise ValueError(str(exc)) from exc


@wraps(MediaConnector.fetch_image_async)
async def _fetch_image_async(
    self: MediaConnector,
    image_url: str,
    *,
    image_mode: str | None = "RGB",
) -> Any:
    image_io = _make_image_io(self, image_mode)
    try:
        return await self.load_from_url_async(
            image_url,
            image_io,
            fetch_timeout=envs.VLLM_IMAGE_FETCH_TIMEOUT,
        )
    except UnidentifiedImageError as exc:
        raise ValueError(str(exc)) from exc


@wraps(MediaConnector.fetch_video)
def _fetch_video(
    self: MediaConnector,
    video_url: str,
    *,
    image_mode: str | None = "RGB",
    video_processor: str | None = None,
) -> Any:
    image_io = _make_image_io(self, image_mode)
    video_io_kwargs = dict(self.media_io_kwargs.get("video", {}))
    if "video_backend" not in video_io_kwargs and (
        video_backend := get_video_loader_backend_for_processor(video_processor)
    ):
        video_io_kwargs["video_backend"] = video_backend
    video_io = VideoMediaIO(image_io, **video_io_kwargs)
    return self.load_from_url(
        video_url,
        video_io,
        fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
    )


@wraps(MediaConnector.fetch_video_async)
async def _fetch_video_async(
    self: MediaConnector,
    video_url: str,
    *,
    image_mode: str | None = "RGB",
    video_processor: str | None = None,
) -> Any:
    image_io = _make_image_io(self, image_mode)
    video_io_kwargs = dict(self.media_io_kwargs.get("video", {}))
    if "video_backend" not in video_io_kwargs and (
        video_backend := get_video_loader_backend_for_processor(video_processor)
    ):
        video_io_kwargs["video_backend"] = video_backend
    video_io = VideoMediaIO(image_io, **video_io_kwargs)
    return await self.load_from_url_async(
        video_url,
        video_io,
        fetch_timeout=envs.VLLM_VIDEO_FETCH_TIMEOUT,
    )


def install_media_connector_image_mode_backport() -> None:
    """Install the vLLM #49159 behavior on v0.26.0 only."""

    if not vllm_version_is("0.26.0"):
        return
    if hasattr(MediaConnector, _ORIGINAL_CONNECTOR_METHODS_ATTR):
        return

    setattr(
        MediaConnector,
        _ORIGINAL_CONNECTOR_METHODS_ATTR,
        (
            MediaConnector.fetch_image,
            MediaConnector.fetch_image_async,
            MediaConnector.fetch_video,
            MediaConnector.fetch_video_async,
        ),
    )
    setattr(
        ImageMediaIO,
        _ORIGINAL_CONVERT_IMAGE_MODE_ATTR,
        ImageMediaIO._convert_image_mode,
    )
    ImageMediaIO._convert_image_mode = _convert_image_mode
    MediaConnector.fetch_image = _fetch_image
    MediaConnector.fetch_image_async = _fetch_image_async
    MediaConnector.fetch_video = _fetch_video
    MediaConnector.fetch_video_async = _fetch_video_async


install_media_connector_image_mode_backport()
