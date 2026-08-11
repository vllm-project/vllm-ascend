=== "Pre-built image (recommended)"

    ## 4.1 Use a Pre-built vLLM Ascend Image {: #pre-built-image }

    If your goal is to obtain a reproducible environment as quickly as possible, use a pre-built image.

    The host only needs:

    - a working Ascend driver/firmware;
    - Docker.

    The image provides the CANN userspace, PyTorch/TorchNPU, vLLM, and vLLM Ascend.

    ### 4.1.1 Select an Image {: #pre-built-image-matrix }

{% include "getting_started/image_matrix.inc.md" %}

    Use this matrix during installation to select an image by hardware and container operating system. In the Quickstart, use the complete image name and container command provided for the corresponding hardware path.

    ### 4.1.2 Pull the Image {: #pre-built-image-pull }

    Select an `IMAGE` from the table above that matches your hardware and container operating system.

    For example:

    ```bash
    export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}
    docker pull "$IMAGE"
    docker image inspect "$IMAGE" >/dev/null && echo "Image ready: $IMAGE"
    ```

    When you see:

    ```text
    Image ready: ...
    ```

    This indicates that the image is ready.

    !!! note "Why doesn't Installation maintain every docker run command here?"

        Device mappings and host mounts vary substantially between hardware products, especially for Atlas A3 and Atlas 200I Pro.

        To avoid maintaining duplicate container commands in Installation and the Quickstart, this page covers only: **host ready + image ready**.

        After proceeding to [Quickstart > 3. Select Hardware and Run](quick_start.md#quickstart-hardware), copy the complete `docker run` command for your actual hardware and continue to your first inference.

    ??? "Build a vLLM Ascend image from a Dockerfile"

        To modify the image contents or package local code, build from source:

        ```bash
        git clone --depth 1 --branch {{ vllm_ascend_version }} \
            https://github.com/vllm-project/vllm-ascend.git
        cd vllm-ascend
        ```

        The default `Dockerfile` targets Atlas A2. For other hardware, use the corresponding Dockerfile in the repository, such as an A3, 310P image variant, or 950-series file.

        Example:

        ```bash
        docker build -t vllm-ascend-dev:latest -f Dockerfile .
        ```

    ### 4.1.3 Completion Criteria {: #pre-built-image-complete }

    The pre-built image installation path is complete when `docker image inspect "$IMAGE"` succeeds and prints `Image ready`. Next, proceed to the [Quickstart](quick_start.md) and run your first inference with the complete container command for your hardware.
