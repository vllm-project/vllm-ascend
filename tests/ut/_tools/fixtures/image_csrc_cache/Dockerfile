# CPU-only transport fixture. Synthetic CANN metadata is NOT Ascend validation.
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends g++ && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux /usr/local/Ascend/nnal/atb && \
    printf 'fixture-only\n' > /usr/local/Ascend/ascend-toolkit/latest/x86_64-linux/ascend_toolkit_install.info && \
    touch /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh
