# SPDX-License-Identifier: Apache-2.0
"""MoEOffloadConfig parsing/validation (ascend_config.moe_offload_config)."""

import pytest

ascend_config = pytest.importorskip("vllm_ascend.ascend_config", reason="needs vllm config stack")


@pytest.fixture
def MoEOffloadConfig():
    return ascend_config.MoEOffloadConfig


class TestDefaults:
    def test_empty_config_gets_defaults(self, MoEOffloadConfig):
        c = MoEOffloadConfig()
        assert c.enabled is False
        assert c.slot_count == 2
        assert c.pin_memory is True

    def test_none_config_gets_defaults(self, MoEOffloadConfig):
        c = MoEOffloadConfig(None)
        assert c.enabled is False


class TestLegacyAlias:
    def test_weight_offload_field_alias(self, MoEOffloadConfig):
        # older scripts wrote {"bifrost_config": {"weight_offload": true}};
        # the field must map onto "enabled"
        c = MoEOffloadConfig({"weight_offload": True})
        assert c.enabled is True

    def test_explicit_enabled_wins(self, MoEOffloadConfig):
        c = MoEOffloadConfig({"weight_offload": False, "enabled": True})
        assert c.enabled is True


class TestMerge:
    def test_override_known_keys(self, MoEOffloadConfig):
        c = MoEOffloadConfig({"enabled": True, "slot_count": 6, "pin_memory": False})
        assert c.enabled is True
        assert c.slot_count == 6
        assert c.pin_memory is False

    def test_unknown_key_raises(self, MoEOffloadConfig):
        # strict: typos in additional-config must fail fast, not be ignored
        with pytest.raises(ValueError, match="no attribute"):
            MoEOffloadConfig({"weight_offlad": True})  # typo


class TestValidation:
    def test_slot_count_must_be_positive(self, MoEOffloadConfig):
        with pytest.raises(ValueError, match="slot_count"):
            MoEOffloadConfig({"slot_count": 0})

    def test_slot_count_must_be_int(self, MoEOffloadConfig):
        with pytest.raises(ValueError, match="slot_count"):
            MoEOffloadConfig({"slot_count": "2"})

    def test_pin_memory_must_be_bool(self, MoEOffloadConfig):
        with pytest.raises(TypeError, match="pin_memory"):
            MoEOffloadConfig({"pin_memory": "yes"})

    def test_large_slot_count_accepted(self, MoEOffloadConfig):
        c = MoEOffloadConfig({"slot_count": 8})
        assert c.slot_count == 8


class TestAttributeAccess:
    def test_getattr_unknown_raises(self, MoEOffloadConfig):
        c = MoEOffloadConfig()
        with pytest.raises(AttributeError):
            _ = c.nonexistent_key
