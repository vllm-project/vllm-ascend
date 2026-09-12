# /*Copyright (c) 2025 Huawei Technologies Co., Ltd.
# *
# * Licensed under the OpenSSL license (the "License").  You may not use
# * this file except in compliance with the License.  You can obtain a copy
# * in the file LICENSE in the source distribution or at
# * https://www.openssl.org/source/license.html
# */

"""Regression tests for PP stage-local layer numbering."""

from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (
    KVPoolWorker,
)


def test_reuse_layout_remap_to_stage_local():
    stage_globals = list(range(38, 79))

    local_map, local_independent = KVPoolWorker._remap_layout_to_stage_local(
        {42: 39, 43: 40, 44: 41, 45: 42}, [38], stage_globals
    )

    assert local_map == {4: 1, 5: 2, 6: 3, 7: 4}
    assert local_independent == [0]
    assert KVPoolWorker._remap_layout_to_stage_local({3: 0, 4: 1}, [0], list(range(38))) == ({3: 0, 4: 1}, [0])
