# SPDX-License-Identifier: Apache-2.0

from tests.ut.base import TestBase
from vllm_ascend.worker.worker import validate_profile_memory_shrink


class TestValidateProfileMemoryShrink(TestBase):
    def test_equal_snapshot_is_accepted(self):
        # Under memory-virtualization layers (e.g. HAMi vNPU quotas) the free
        # memory before and after profiling can legitimately be identical
        # (#18343): the assertion must accept an unchanged snapshot.
        validate_profile_memory_shrink(10 * 1024**3, 10 * 1024**3)

    def test_shrunk_memory_is_accepted(self):
        # Normal case: profiling consumes memory, free memory shrinks.
        validate_profile_memory_shrink(60 * 1024**3, 20 * 1024**3)

    def test_grown_memory_still_raises(self):
        # A *larger* post-profile free memory still indicates unstable
        # external memory changes and must keep raising.
        with self.assertRaises(AssertionError, msg="Error in memory profiling"):
            validate_profile_memory_shrink(10 * 1024**3, 20 * 1024**3)
