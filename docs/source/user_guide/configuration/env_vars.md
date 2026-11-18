# Environment Variables

vllm-ascend uses the following environment variables to configure the system:

For `ASCEND_RT_VISIBLE_DEVICES` and its interaction with `--device-ids`, see
[NPU Device Selection](device_selection.md). The visibility variable is a runtime
setting and remains supported alongside explicit device selection.

**Note:** Some environment variables are being migrated to `--additional-config` options. These environment variables are still supported during the migration period, and it is recommended to use `--additional-config` for new deployments. See [Additional Configuration](additional_config.md) for details.

{{ include_code('vllm_ascend/envs.py', start_after='begin-env-vars-definition', end_before='end-env-vars-definition') }}
