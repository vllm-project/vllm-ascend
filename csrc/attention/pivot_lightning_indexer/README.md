# PIVOT LightningIndexer

`PivotLightningIndexer` is a separate custom operator derived from the native
`LightningIndexer`. It deliberately has its own ACLNN and PyTorch entry points;
enabling PIVOT does not replace or patch the native operator.

On Ascend 910B/910_93, supported single-request causal shapes pool consecutive
query rows in groups of up to four, execute one proxy LightningIndexer row per
group, then restore each row's causal local window. Unsupported shapes execute
the unchanged native algorithm inside this operator. Other architectures are
not built for this experimental operator.

The vLLM integration is opt-in through
`additional_config.enable_pivot_lightning_indexer`. The default is `false`.
