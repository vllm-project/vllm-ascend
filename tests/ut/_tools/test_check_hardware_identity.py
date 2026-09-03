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
#

from textwrap import dedent

import pytest

from tools.check_hardware_identity import check_source, main


def _codes(source: str, path: str = "vllm_ascend/business_logic.py") -> set[str]:
    return {violation.code for violation in check_source(dedent(source), path)}


@pytest.mark.parametrize(
    ("source", "expected_code"),
    [
        (
            """
            from vllm_ascend.utils import is_310p

            if is_310p():
                pass
            """,
            "HDI001",
        ),
        (
            """
            from vllm_ascend.device.device_config import is_950 as on_a5

            if on_a5():
                pass
            """,
            "HDI001",
        ),
        (
            """
            import vllm_ascend.device.device_config as dc

            kind = dc.get_ascend_device_type()
            """,
            "HDI001",
        ),
        (
            """
            import vllm_ascend.utils as ascend_utils

            enabled = ascend_utils.is_310p()
            """,
            "HDI001",
        ),
        (
            """
            from vllm_ascend.device.hardware import AscendDeviceType as DT

            enabled = kind == DT.A5
            """,
            "HDI001",
        ),
        (
            """
            from vllm_ascend.device.hardware import AscendDeviceType as DT

            enabled = DT.A5 is kind
            """,
            "HDI001",
        ),
        (
            """
            from vllm_ascend.device.hardware import AscendDeviceType

            enabled = kind in {AscendDeviceType.A2, AscendDeviceType.A3}
            """,
            "HDI001",
        ),
        (
            """
            from vllm_ascend.device.hardware import AscendDeviceType

            match kind:
                case AscendDeviceType.A5:
                    pass
            """,
            "HDI001",
        ),
        (
            """
            import torch_npu

            runtime = torch_npu.npu.get_soc_version()
            """,
            "HDI002",
        ),
        (
            """
            import os

            build_target = os.getenv("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            build_target = os.getenv(key="SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            read_env = os.getenv
            build_target = read_env(key="SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            read_env = os.getenv

            def read_build_target(read_env=read_env):
                return read_env("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            read_env = os.getenv

            def read_build_target(*, read_env=read_env):
                return read_env("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            build_target = (read_env := os.getenv)("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            build_target = (environment := os.environ).get("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            build_target = (read_env := os.environ.get)("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            class Configuration:
                if False:
                    os = fake
                build_target = os.getenv("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            class Configuration:
                if use_fake_environment:
                    os = fake
                build_target = os.getenv("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            class Configuration:
                build_target = os.getenv("SOC_VERSION")
                os = fake
            """,
            "HDI002",
        ),
        (
            """
            from os import *

            build_target = getenv("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            read_env = other = os.getenv
            build_target = other("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            read_env = (other := os.getenv)
            build_target = other("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            def read_build_target(read_env=os.getenv):
                return read_env("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os as operating_system

            build_target = operating_system.environ.get("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            from os import environ as process_environment

            build_target = process_environment["SOC_VERSION"]
            """,
            "HDI002",
        ),
        (
            """
            import os

            read_env = os.getenv
            build_target = read_env("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            environment = os.environ
            build_target = environment["SOC_VERSION"]
            """,
            "HDI002",
        ),
        (
            """
            import os

            environment = os.environ
            build_target = environment.get("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            read_env: object = os.environ.get
            build_target = read_env("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            def read_build_target():
                return read_env("SOC_VERSION")

            import os
            read_env = os.getenv
            """,
            "HDI002",
        ),
        (
            """
            def read_build_target():
                return read_env("SOC_VERSION")

            from os import getenv as read_env
            """,
            "HDI002",
        ),
        (
            """
            def outer():
                def read_build_target():
                    return read_env("SOC_VERSION")

                import os
                read_env = os.getenv
                return read_build_target()
            """,
            "HDI002",
        ),
        (
            """
            import os.path

            build_target = os.getenv("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            read_env, other = os.getenv, object()
            build_target = read_env("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            import os

            class Outer:
                os = fake

                class Inner:
                    build_target = os.getenv("SOC_VERSION")
            """,
            "HDI002",
        ),
        (
            """
            device_kind = get_device_config()._device_type
            """,
            "HDI003",
        ),
        (
            """
            import vllm_ascend.device.device_config as dc

            predicate = getattr(dc, "is_950")
            """,
            "HDI004",
        ),
        (
            """
            predicate = globals()["is_310p"]
            """,
            "HDI004",
        ),
        (
            """
            from vllm_ascend.device.device_config import *
            """,
            "HDI001",
        ),
        (
            """
            SOC_VERSION_INFERENCE_SERIES = ["Ascend310P3"]
            """,
            "HDI002",
        ),
        (
            """
            if torch.npu.get_device_name().startswith("Ascend950"):
                enable_fast_path()
            """,
            "HDI005",
        ),
        (
            """
            if device_name in {"Ascend310P3", "Ascend310P5"}:
                use_host_staging()
            """,
            "HDI005",
        ),
        (
            """
            match hardware_family:
                case "A3":
                    enable_fullmesh()
            """,
            "HDI005",
        ),
        (
            """
            A5_SOCS = {"Ascend950"}

            if device_name in A5_SOCS:
                enable_fast_path()
            """,
            "HDI005",
        ),
        (
            """
            predicate = globals().get("is_310p")
            """,
            "HDI004",
        ),
        (
            """
            def is_a5_bf16_kv_enabled():
                return True
            """,
            "HDI001",
        ),
        (
            """
            if is_950_variant():
                enable_fast_path()
            """,
            "HDI001",
        ),
        (
            """
            if is_310p3vir01():
                use_host_staging()
            """,
            "HDI001",
        ),
    ],
)
def test_rejects_explicit_hardware_identity(source, expected_code):
    assert expected_code in _codes(source)


@pytest.mark.parametrize(
    "source",
    [
        """
        from vllm_ascend.device.hardware_profile import HardwareCapability
        from vllm_ascend.device.hardware_profile import get_current_hardware_profile

        enabled = get_current_hardware_profile().supports(
            HardwareCapability.FP8_ATTENTION
        )
        """,
        """
        from vllm_ascend.device.hardware_profile import AttentionBackendFamily
        from vllm_ascend.device.hardware_profile import get_current_hardware_profile

        family = get_current_hardware_profile().attention_backend_family
        enabled = family is AttentionBackendFamily.DENSE_MLA_SFA_DSA
        """,
        """
        enabled = current_platform.device_type == "npu"
        fallback = config.device_type == "cuda"
        """,
        """
        # is_310p and AscendDeviceType are documentation here.
        note = "Do not branch on get_ascend_device_type or Ascend950"
        """,
        """
        class Ascend310PWorker:
            pass

        def get_310p_rope_state():
            pass
        """,
        """
        from vllm_ascend.device.device_config import check_ascend_device_type

        check_ascend_device_type()
        """,
        """
        from vllm_ascend.utils import *

        enable_custom_op()
        """,
        """
        import os

        read_env = os.getenv
        environment = os.environ
        cache_root = read_env("VLLM_CACHE_ROOT")
        executable_path = environment.get("PATH")
        """,
        """
        import os

        read_env = os.getenv

        def read_setting(read_env):
            return read_env("SOC_VERSION")
        """,
        """
        import os

        values = [os.getenv("SOC_VERSION") for os in providers]
        """,
        """
        import os

        class Configuration:
            os = fake
            build_target = os.getenv("SOC_VERSION")
        """,
    ],
)
def test_accepts_profile_queries_and_non_identity_device_names(source):
    assert not _codes(source)


@pytest.mark.parametrize(
    "path",
    [
        "vllm_ascend/device/device_config.py",
        "vllm_ascend/device/hardware.py",
        "vllm_ascend/device/hardware_profile.py",
    ],
)
def test_allows_identity_only_in_exact_hal_boundary_files(path):
    source = """
        from vllm_ascend.device.hardware import AscendDeviceType

        enabled = get_ascend_device_type() is AscendDeviceType.A5
    """
    assert not _codes(source, path)


def test_does_not_allow_neighboring_device_modules():
    source = """
        from vllm_ascend.device.hardware import AscendDeviceType

        enabled = kind == AscendDeviceType.A5
    """
    assert "HDI001" in _codes(source, "vllm_ascend/device/backend.py")


def test_skips_tests_that_construct_profiles_from_device_types():
    source = """
        from vllm_ascend.device.hardware import AscendDeviceType

        profile = get_hardware_profile(AscendDeviceType.A3)
    """
    assert not _codes(source, "tests/ut/device/test_hardware_profile.py")


def test_allows_the_existing_lazy_soc_version_environment_declaration():
    source = """
        import os

        env_variables: dict[str, object] = {
            "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
        }
    """
    assert not _codes(source, "vllm_ascend/envs.py")


def test_envs_file_exception_does_not_cover_an_arbitrary_neighboring_dict():
    source = """
        import os

        env_variables: dict[str, object] = {}
        other_environment_variables = {
            "SOC_VERSION": lambda: os.getenv("SOC_VERSION", None),
        }
    """
    assert "HDI002" in _codes(source, "vllm_ascend/envs.py")


@pytest.mark.parametrize(
    "path",
    [
        "vllm_ascend/_build_info.py",
        "vllm_ascend/_cann_ops_custom/vendor/generated_op.py",
    ],
)
def test_skips_generated_build_outputs_outside_python_business_logic(path):
    source = "selected = get_soc_version() == 'Ascend950'"
    assert not _codes(source, path)


def test_envs_file_exception_does_not_cover_a_soc_identity_branch():
    source = """
        import os

        if os.getenv("SOC_VERSION") == "Ascend950":
            enable_fast_path()
    """
    assert "HDI002" in _codes(source, "vllm_ascend/envs.py")


def test_reports_parse_errors_in_scanned_production_files():
    violations = check_source("if True print('broken')", "vllm_ascend/broken.py")
    assert [violation.code for violation in violations] == ["HDI000"]


@pytest.mark.parametrize(
    "identity",
    [
        "910b",
        "910c",
        "910",
        "310p",
        "Ascend910",
        "Ascend910B1",
        "Ascend910B2",
        "Ascend910B2C",
        "Ascend910B3",
        "Ascend910B4",
        "Ascend910B4-1",
        "Ascend910_9391",
        "Ascend910_9381",
        "Ascend910_9372",
        "Ascend910_9392",
        "Ascend910_9382",
        "Ascend910_9362",
        "Ascend310P1",
        "Ascend310P3",
        "Ascend310P5",
        "Ascend310P7",
        "Ascend310P3Vir01",
        "Ascend310P3Vir02",
        "Ascend310P3Vir04",
        "Ascend310P3Vir08",
        "Ascend950",
        "_310P",
        "A2",
        "A3",
        "A5",
        "Ascend910C1",
    ],
)
def test_rejects_known_and_forward_compatible_device_literals(identity: str):
    assert "HDI005" in _codes(f"selected = device_name == {identity!r}")


def test_default_cli_scan_covers_the_complete_production_tree():
    assert main([]) == 0
