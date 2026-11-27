# SPDX-License-Identifier: Apache-2.0
"""_resolved_kv_layout: CacheConfig wins over empty env (the Ascend
layout-consumer patch's correctness proof).

The fallback is patched at the consumer-visible attribute
(vllm.envs.VLLM_KV_CACHE_LAYOUT on the attention module): patching
os.environ instead is fragile in a shared process, because ANY earlier
test that patches the lazy vllm.envs attribute (unittest.mock /
pytest.MonkeyPatch save the __getattr__ value and materialize it on
undo) leaves a real module attribute behind that shadows every later
environ read.
"""

from types import SimpleNamespace
from unittest.mock import patch


def _config(kv_cache_layout):
    return SimpleNamespace(cache_config=SimpleNamespace(kv_cache_layout=kv_cache_layout))


class TestResolvedKvLayout:
    """_resolved_kv_layout: CacheConfig.kv_cache_layout wins over the raw
    env; the env is the fallback for pre-config contexts."""

    def test_cache_config_wins_over_env(self):
        from vllm_ascend.attention import attention_v1

        mock_config = _config("LBHNC")
        with (
            patch(
                "vllm_ascend.attention.attention_v1.get_current_vllm_config",
                return_value=mock_config,
            ),
            patch.object(attention_v1.envs_vllm, "VLLM_KV_CACHE_LAYOUT", "HND"),
        ):
            assert attention_v1._resolved_kv_layout() == "LBHNC"

    def test_env_fallback_when_config_empty(self):
        from vllm_ascend.attention import attention_v1

        mock_config = _config(None)
        with (
            patch(
                "vllm_ascend.attention.attention_v1.get_current_vllm_config",
                return_value=mock_config,
            ),
            patch.object(attention_v1.envs_vllm, "VLLM_KV_CACHE_LAYOUT", "HND"),
        ):
            assert attention_v1._resolved_kv_layout() == "HND"

    def test_none_when_both_empty(self):
        from vllm_ascend.attention import attention_v1

        mock_config = _config(None)
        with (
            patch(
                "vllm_ascend.attention.attention_v1.get_current_vllm_config",
                return_value=mock_config,
            ),
            patch.object(attention_v1.envs_vllm, "VLLM_KV_CACHE_LAYOUT", None),
        ):
            assert attention_v1._resolved_kv_layout() is None

    def test_config_exception_falls_back_to_env(self):
        from vllm_ascend.attention import attention_v1

        with (
            patch(
                "vllm_ascend.attention.attention_v1.get_current_vllm_config",
                side_effect=RuntimeError("no config context"),
            ),
            patch.object(attention_v1.envs_vllm, "VLLM_KV_CACHE_LAYOUT", "HND"),
        ):
            assert attention_v1._resolved_kv_layout() == "HND"
