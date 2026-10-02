# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Glm5NextProcessor multimodal token counting."""

import unittest
from types import SimpleNamespace

from vllm_ascend.models.glm5next.processor import Glm5NextProcessor


def _fake_processor(merge_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        image_processor=SimpleNamespace(
            merge_size=merge_size,
            get_number_of_image_patches=lambda height, width, _kwargs: height * width,
        ),
        video_processor=SimpleNamespace(
            merge_size=merge_size,
            get_number_of_video_patches=lambda frames, height, width, _kwargs: frames * height * width,
        ),
    )


class TestGlm5NextProcessorTokenCount(unittest.TestCase):
    def test_video_only_request_counts_tokens(self):
        # Regression: the video branch read merge_size, which was only bound
        # inside the `if image_sizes is not None:` branch, so any video-only
        # call raised NameError before the fix.
        data = Glm5NextProcessor._get_num_multimodal_tokens(
            _fake_processor(merge_size=2),
            video_sizes=[(2, 4, 6)],
            merge_size=2,
        )
        self.assertEqual(data["num_video_tokens"], [12])  # 2*4*6 patches // 2**2

    def test_image_only_request_counts_tokens(self):
        data = Glm5NextProcessor._get_num_multimodal_tokens(
            _fake_processor(merge_size=2),
            image_sizes=[(4, 6)],
            merge_size=2,
        )
        self.assertEqual(data["num_image_patches"], [24])
        self.assertEqual(data["num_image_tokens"], [6])  # 24 patches // 2**2


if __name__ == "__main__":
    unittest.main()
