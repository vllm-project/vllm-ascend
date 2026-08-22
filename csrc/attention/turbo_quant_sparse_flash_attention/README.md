# TurboQuant Sparse Flash Attention Source

The operator implementation in `op_host` and `op_kernel` is synchronized from
the CANN `ops-transformer` contribution below:

- Pull request: <https://gitcode.com/cann/ops-transformer/pull/9890>
- Source commit: `4043903c647d76b829be4cefe61249a9c6764032`
- Source path: `experimental/attention/turbo_quant_sparse_flash_attention`

Those implementation files are licensed under the CANN Open Software License
Agreement Version 2.0. The vLLM-Ascend build
integration and `turbo_quant_sparse_flash_attention_torch_adpt.h` remain
repository-local integration code and are not copied from the CANN operator
repository.
