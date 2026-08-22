# TurboQuant Compress Latent Source

The operator implementation in `op_host` and `op_kernel` is synchronized from
the CANN `ops-nn` contribution below:

- Pull request: <https://gitcode.com/cann/ops-nn/pull/8222>
- Source commit: `02b3c80a76a9f55fd42d48637dd098dbba36468b`
- Source path: `experimental/quant/turbo_quant_compress_latent`

Those implementation files are licensed under the CANN Open Software License
Agreement Version 2.0. The vLLM-Ascend build
integration and `turbo_quant_compress_latent_torch_adpt.h` remain repository-local
integration code and are not copied from the CANN operator repository.
