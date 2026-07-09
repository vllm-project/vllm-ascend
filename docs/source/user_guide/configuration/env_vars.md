# Environment Variables

vllm-ascend uses the following environment variables to configure the system:

**Note:** Some environment variables are being migrated to `--additional-config` options. These environment variables are still supported during the migration period, and it is recommended to use `--additional-config` for new deployments. See [Additional Configuration](additional_config.md) for details.

:::{literalinclude} ../../../../vllm_ascend/envs.py
:language: python
:start-after: begin-env-vars-definition
:end-before: end-env-vars-definition
:::
