# Environment Variables

vllm-ascend uses the following environment variables to configure the system:

**Note:** Some environment variables are being migrated to `--additional-config` options. These environment variables are still supported during the migration period, and it is recommended to use `--additional-config` for new deployments. See [Additional Configuration](additional_config.md) for details.

{{ include_code('vllm_ascend/envs.py', start_after='begin-env-vars-definition', end_before='end-env-vars-definition') }}
