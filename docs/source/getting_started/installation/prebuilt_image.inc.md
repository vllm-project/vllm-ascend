=== "Prebuilt image"

    <span id="installation-prebuilt-image"></span>

    The host needs only a working Ascend driver and firmware, plus Docker. The image includes the CANN user-space environment, PyTorch/TorchNPU, vLLM, and vLLM Ascend.

    Choose an official image for the fastest setup, or build an image manually when you need to customize it.

    <span id="installation-ascend-docker-runtime"></span>

    **Use Ascend Docker Runtime.** The runtime mounts the NPU devices and driver paths of the host, and a container then selects its NPUs with `-e ASCEND_VISIBLE_DEVICES=<ids>`. Install it as `root` on every node that runs containers, following the [MindCluster installation guide](https://www.hiascend.com/document/detail/en/mindcluster/latest/):

    ```bash
    chmod u+x Ascend-docker-runtime_<version>_linux-<arch>.run
    ./Ascend-docker-runtime_<version>_linux-<arch>.run --install
    systemctl daemon-reload && systemctl restart docker
    ```

    Confirm that the runtime is the default:

    {% raw %}

    ```bash
    docker info --format 'default={{.DefaultRuntime}}'
    ```

    {% endraw %}

    The output must be `default=ascend`. If it prints `runc`, check the `Runtimes:` line of `docker info`:

    1. If `ascend` is listed, either add `--runtime ascend` to `docker run`, or set `"default-runtime": "ascend"` in `/etc/docker/daemon.json` and restart Docker.
    2. If `ascend` is missing, rerun the installation and restart Docker.

    The runtime mounts `/etc/ascend-docker-runtime.d/base.list` into every container, read-only. To mount an extra host path in every container, add it to `base.list`, one path per line. To mount a path only in selected containers, create another lowercase `.list` file in that directory and name it in `ASCEND_RUNTIME_MOUNTS`, for example `-e ASCEND_RUNTIME_MOUNTS=mylist,base`: that variable replaces the default list, so include `base`; otherwise the container loses the driver libraries and `npu-smi`.

    === "Select an official image"

        <span id="installation-prebuilt-image-selection"></span>

        Select your hardware and operating system, then pull the official image, start the container, and verify the environment. The commands select NPUs with `ASCEND_VISIBLE_DEVICES` and need [Ascend Docker Runtime](#installation-ascend-docker-runtime) on the host. The examples use the device ids of a full node; on a machine with fewer devices, pass only the devices it has.

        ??? note "Official vLLM Ascend images"

            | Image name | Hardware | OS |
            | --- | --- | --- |
            | `vllm-ascend:{{ vllm_ascend_version }}` | Ascend A2 series products | Ubuntu |
            | `vllm-ascend:{{ vllm_ascend_version }}-openeuler` | Ascend A2 series products | openEuler |
            | `vllm-ascend:{{ vllm_ascend_version }}-a3` | Ascend A3 series products | Ubuntu |
            | `vllm-ascend:{{ vllm_ascend_version }}-a3-openeuler` | Ascend A3 series products | openEuler |
            | `vllm-ascend:{{ vllm_ascend_version }}-310p` | Atlas 300I DUO / Atlas 200I Pro | Ubuntu |
            | `vllm-ascend:{{ vllm_ascend_version }}-310p-openeuler` | Atlas 300I DUO / Atlas 200I Pro | openEuler |
            | `vllm-ascend:{{ vllm_ascend_version }}-a5` | 950DT Products | Ubuntu |
            | `vllm-ascend:{{ vllm_ascend_version }}-a5-openeuler` | 950DT Products | openEuler |

{% filter indent(8, true) %}{% include "getting_started/quick_start/ascend_image/atlas-a2.inc.md" %}{% endfilter %}

{% filter indent(12, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

{% filter indent(8, true) %}{% include "getting_started/quick_start/ascend_image/atlas-a3.inc.md" %}{% endfilter %}

{% filter indent(12, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

{% filter indent(8, true) %}{% include "getting_started/quick_start/ascend_image/atlas-300i-duo.inc.md" %}{% endfilter %}

{% filter indent(12, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

{% filter indent(8, true) %}{% include "getting_started/quick_start/ascend_image/atlas-200i-pro.inc.md" %}{% endfilter %}

{% filter indent(12, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

{% filter indent(8, true) %}{% include "getting_started/quick_start/ascend_image/atlas-950dt.inc.md" %}{% endfilter %}

{% filter indent(12, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

    === "Build an image manually"

        <span id="installation-prebuilt-image-build"></span>

        Run the command for your hardware and operating system:

        === "A2"

            === "Ubuntu"

                ```bash
                git clone --depth 1 --branch {{ vllm_ascend_version }} \
                    https://github.com/vllm-project/vllm-ascend.git
                cd vllm-ascend
                docker build -t vllm-ascend-dev:latest -f Dockerfile .
                ```

            === "openEuler"

                ```bash
                git clone --depth 1 --branch {{ vllm_ascend_version }} \
                    https://github.com/vllm-project/vllm-ascend.git
                cd vllm-ascend
                docker build -t vllm-ascend-dev:latest -f Dockerfile.openEuler .
                ```

        === "A3"

            === "Ubuntu"

                ```bash
                git clone --depth 1 --branch {{ vllm_ascend_version }} \
                    https://github.com/vllm-project/vllm-ascend.git
                cd vllm-ascend
                docker build -t vllm-ascend-dev:latest -f Dockerfile.a3 .
                ```

            === "openEuler"

                ```bash
                git clone --depth 1 --branch {{ vllm_ascend_version }} \
                    https://github.com/vllm-project/vllm-ascend.git
                cd vllm-ascend
                docker build -t vllm-ascend-dev:latest -f Dockerfile.a3.openEuler .
                ```

        === "Atlas 300I DUO / Atlas 200I Pro"

            === "Ubuntu"

                ```bash
                git clone --depth 1 --branch {{ vllm_ascend_version }} \
                    https://github.com/vllm-project/vllm-ascend.git
                cd vllm-ascend
                docker build -t vllm-ascend-dev:latest -f Dockerfile.310p .
                ```

            === "openEuler"

                ```bash
                git clone --depth 1 --branch {{ vllm_ascend_version }} \
                    https://github.com/vllm-project/vllm-ascend.git
                cd vllm-ascend
                docker build -t vllm-ascend-dev:latest -f Dockerfile.310p.openEuler .
                ```

        === "950DT Products"

            === "Ubuntu"

                ```bash
                git clone --depth 1 --branch {{ vllm_ascend_version }} \
                    https://github.com/vllm-project/vllm-ascend.git
                cd vllm-ascend
                docker build -t vllm-ascend-dev:latest -f Dockerfile.a5 .
                ```

            === "openEuler"

                ```bash
                git clone --depth 1 --branch {{ vllm_ascend_version }} \
                    https://github.com/vllm-project/vllm-ascend.git
                cd vllm-ascend
                docker build -t vllm-ascend-dev:latest -f Dockerfile.a5.openEuler .
                ```
