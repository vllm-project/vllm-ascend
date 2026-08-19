    The goal of Installation is not to run a specific LLM, but to confirm that the software stack and NPU backend are ready.

    The [Quickstart](quick_start.md) covers the model-level inference smoke test.

    #### 1. Check the Devices

    ```bash
    npu-smi info

    printf 'ASCEND_VISIBLE_DEVICES=%s\n' \
        "${ASCEND_VISIBLE_DEVICES-<unset>}"

    printf 'ASCEND_RT_VISIBLE_DEVICES=%s\n' \
        "${ASCEND_RT_VISIBLE_DEVICES-<unset>}"
    ```

    If Docker `--device` already restricts the visible devices, you generally do not need to set both visibility variables as well.

    #### 2. Check the Core Software and NPU Backend

    ```bash
    PYTHONPATH= pip check
    ```

    Then run:

    ```bash
    python - <<'PY'
    from importlib.metadata import PackageNotFoundError, entry_points, version

    import torch
    import torch_npu  # noqa: F401
    import vllm  # noqa: F401
    import vllm_ascend  # noqa: F401

    packages = (
        "torch",
        "torch-npu",
        "triton",
        "triton-ascend",
        "vllm",
        "vllm-ascend",
    )

    for package in packages:
        try:
            package_version = version(package)
        except PackageNotFoundError:
            package_version = "not installed"
        print(f"{package}: {package_version}")

    assert hasattr(torch, "npu"), "torch.npu is unavailable"
    assert torch.npu.is_available(), "No Ascend NPU is available to PyTorch"

    lhs = torch.ones((2, 2), device="npu")
    result = (lhs @ lhs).cpu()
    assert torch.equal(result, torch.full((2, 2), 2.0)), result

    plugins = entry_points(group="vllm.platform_plugins")
    ascend_plugins = [ep for ep in plugins if ep.name == "ascend"]
    assert ascend_plugins, "vLLM Ascend platform plugin is not registered"

    print("NPU tensor operation: PASS")
    print("vLLM Ascend plugin:", ascend_plugins[0].value)
    print("Installation verification: PASS")
    PY
    ```

    The minimum criteria for installation readiness are:

    - the host / current runtime environment can detect the NPU;
    - `torch.npu.is_available()` succeeds;
    - the simple NPU tensor operation passes;
    - the vLLM Ascend plugin is registered;
    - there are no unresolved core software dependency conflicts.

    After meeting these criteria, proceed to the [Quickstart](quick_start.md) and run the first model for your hardware.
