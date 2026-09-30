??? tip "If image downloads are slow"

    vLLM Ascend images are downloaded from `quay.io` by default. If direct access is slow, use one of the following registry mirrors to accelerate the download.

    For example, the original image address is:

    ```text
    quay.io/ascend/vllm-ascend:<TAG>
    ```

    You can replace it with:

    ```text
    # Replace with tag you want to pull
    TAG={{ vllm_ascend_version }}
    # use
    docker pull m.daocloud.io/quay.io/ascend/vllm-ascend:$TAG
    # or
    docker pull quay.nju.edu.cn/ascend/vllm-ascend:$TAG
    ```

    Replace only the registry prefix and preserve the complete original image tag, including suffixes such as `-a3`, `-310p`, `-950dt`, and `-openeuler`.
