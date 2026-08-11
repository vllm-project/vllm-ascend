    If no working CANN environment is available, first install the CANN version that matches both the target hardware and the software stack on this page.

    Open the following official resources:

    - [CANN Community Edition Download Center](https://www.hiascend.com/cann/download)
    - [CANN Software Installation Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0000.html)

    Based on `{{ release_cann_version }}`, the target hardware, CPU architecture, and operating system, obtain the following three types of `.run` installer:

    - CANN Toolkit;
    - the operator package for the target hardware;
    - an NNAL package at the same version as the other two packages.

    !!! important "Confirm NNAL separately"

        The download page or quick installation path may primarily display the Toolkit and operator package, without placing NNAL in the same prominent location. vLLM Ascend requires NNAL/ATB at runtime. Do not skip NNAL simply because the Toolkit installation succeeded.

        The NNAL `.run` package is typically named `Ascend-cann-nnal_...run`. For installation options, refer to the
        [CANN installation parameter reference](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/softwareinst/instg/instg_0043.html).

    !!! important "Do not reuse operator packages across hardware products"

        Atlas A2, Atlas A3, Atlas 300I DUO / Atlas 200I Pro, and Atlas 950DT must use their respective matching operator packages. All three package types must use the same CANN version.

    After downloading the packages, replace the following three variables with the absolute paths to the actual files, then run the installation commands:

    ```bash
    CANN_TOOLKIT_RUN=/absolute/path/to/toolkit.run
    CANN_OPS_RUN=/absolute/path/to/hardware-ops.run
    CANN_NNAL_RUN=/absolute/path/to/nnal.run

    test -f "$CANN_TOOLKIT_RUN"
    test -f "$CANN_OPS_RUN"
    test -f "$CANN_NNAL_RUN"

    chmod +x "$CANN_TOOLKIT_RUN" "$CANN_OPS_RUN" "$CANN_NNAL_RUN"

    "$CANN_TOOLKIT_RUN" --quiet --full
    "$CANN_OPS_RUN" --quiet --install
    "$CANN_NNAL_RUN" --quiet --install
    ```

    After installation, load and check the Toolkit and NNAL:

    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    source /usr/local/Ascend/nnal/atb/set_env.sh

    export ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest}"

    test -f /usr/local/Ascend/nnal/atb/set_env.sh
    find /usr/local/Ascend/nnal -name libatb.so -print -quit | grep -q .
    npu-smi info
    ```

    If you use a non-default installation path, load the `set_env.sh` from the corresponding location. If the NNAL check fails, install the matching NNAL version before continuing with the vLLM Ascend installation.

    ??? tip "CANN package download fails"

        Check the network, proxy/mirror, and disk space in the download environment. Also reconfirm the CANN version, hardware package, and CPU architecture.

    ??? note "Conda CANN"

        If you use CANN Conda packages released by Huawei, follow the complete Conda installation process. Mixing a Conda Toolkit with `.run` operator/NNAL installation methods is not recommended.
