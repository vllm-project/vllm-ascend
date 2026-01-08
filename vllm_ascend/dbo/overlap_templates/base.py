class UbatchOverlapBaseTemplate:

    # Custom hooks for dbo overlap policy
    # Users can control the following aspects by implementing these hooks:
    # (1) kernel launching orders along two streams
    # (2) which kernel to overlap and how to overlap by adding record/wait events
    def dbo_mla_preprocess_hook(self, is_record):
        pass

    def dbo_linear_row_hook(self, is_record):
        pass

    def dbo_linear_column_hook(self, is_record):
        pass

    def dbo_moe_prepare_hook(self, is_record):
        pass

    def dbo_moe_finalize_hook(self, is_record):
        pass
