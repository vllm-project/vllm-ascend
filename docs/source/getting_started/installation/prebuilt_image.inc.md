=== "Prebuilt image"

    <span id="installation-prebuilt-image"></span>

    The host needs only a working Ascend driver and firmware, plus Docker. The image includes the CANN user-space environment, PyTorch/TorchNPU, vLLM, and vLLM Ascend.

    Choose an official image for the fastest setup, or build an image manually when you need to customize it.

    === "Select an official image"

        <span id="installation-prebuilt-image-selection"></span>

        Select your hardware and operating system, then pull the official image, start the container, and verify the environment.

        ??? note "Official vLLM Ascend images"

{% filter indent(12, true) %}{% include "getting_started/quick_start/ascend_image/atlas-a2.inc.md" %}{% endfilter %}

{% filter indent(16, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

{% filter indent(12, true) %}{% include "getting_started/quick_start/ascend_image/atlas-a3.inc.md" %}{% endfilter %}

{% filter indent(16, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

{% filter indent(12, true) %}{% include "getting_started/quick_start/ascend_image/atlas-300i-duo.inc.md" %}{% endfilter %}

{% filter indent(16, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

{% filter indent(12, true) %}{% include "getting_started/quick_start/ascend_image/atlas-200i-pro.inc.md" %}{% endfilter %}

{% filter indent(16, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

{% filter indent(12, true) %}{% include "getting_started/quick_start/ascend_image/atlas-950dt.inc.md" %}{% endfilter %}

{% filter indent(16, true) %}{% include "getting_started/quick_start/ascend_image/verify_container.inc.md" %}{% endfilter %}

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

        === "950DT"

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
