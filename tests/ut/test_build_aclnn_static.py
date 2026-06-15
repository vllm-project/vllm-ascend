from pathlib import Path


def test_build_aclnn_accepts_setup_py_soc_aliases():
    repo_root = Path(__file__).resolve().parents[2]
    script = (repo_root / "csrc" / "build_aclnn.sh").read_text()

    # setup.py may pass short aliases from vllm_ascend.envs.SOC_VERSION.
    for soc_alias in ("310p", "910b", "910c", "950"):
        assert soc_alias in script

    assert 'SOC_ARG="ascend310p"' in script
    assert 'SOC_ARG="ascend910b"' in script
    assert 'SOC_ARG="ascend910_93"' in script
    assert 'SOC_ARG="ascend950"' in script
