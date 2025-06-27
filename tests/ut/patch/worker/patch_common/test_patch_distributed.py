from tests.ut.base import TestBase


class TestPatchDistributed(TestBase):

    def test_GroupCoordinator_patched(self):
        from vllm.distributed.parallel_state import GroupCoordinator

        from vllm_ascend.patch.worker.patch_common.patch_distributed import \
            GroupCoordinatorPatch

        self.assertIs(GroupCoordinator, GroupCoordinatorPatch)
