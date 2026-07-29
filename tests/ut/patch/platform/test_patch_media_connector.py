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

import asyncio
import base64
from io import BytesIO

from PIL import Image
from vllm.multimodal.media.connector import MediaConnector

import vllm_ascend.patch.platform.patch_media_connector  # noqa: F401


def _rgba_data_url() -> str:
    image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    image.putpixel((2, 2), (0, 0, 0, 255))
    buffer = BytesIO()
    image.save(buffer, "PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_media_io_image_mode_override_preserves_original_mode():
    data_url = _rgba_data_url()

    default_image = MediaConnector().fetch_image(data_url)
    assert default_image.mode == "RGB"
    assert default_image.getpixel((0, 0)) == (255, 255, 255)

    connector = MediaConnector(
        media_io_kwargs={
            "image": {
                "image_mode": None,
            },
        },
    )
    images = (
        connector.fetch_image(data_url),
        asyncio.run(connector.fetch_image_async(data_url)),
    )
    for image in images:
        assert image.mode == "RGBA"
        assert image.getpixel((0, 0)) == (0, 0, 0, 0)
        assert image.getpixel((2, 2)) == (0, 0, 0, 255)
