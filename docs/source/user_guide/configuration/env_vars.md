# Environment Variables

vllm-ascend uses the following environment variables to configure the system:

**Note:** Seven former environment variables have been removed and replaced by
`--additional-config` options. See
[Additional Configuration](additional_config.md) for the migration mapping.

{{ include_code('vllm_ascend/envs.py', start_after='begin-env-vars-definition', end_before='end-env-vars-definition') }}
