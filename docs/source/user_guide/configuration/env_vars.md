# Environment Variables

vllm-ascend uses the following environment variables to configure the system:

**Note:** Some environment variables have been migrated to `additional_config` configuration options. These environment variables are still supported for backward compatibility but are marked as deprecated. It is recommended to use `--additional-config` instead. See [Additional Configuration](additional_config.md) for details.

:::{literalinclude} ../../../../vllm_ascend/envs.py
:language: python
:start-after: begin-env-vars-definition
:end-before: end-env-vars-definition
:::
