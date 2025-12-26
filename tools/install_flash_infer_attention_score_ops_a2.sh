cd /vllm-workspace
# download fused_infer_attention_score related source files
wget fused_infer_attention_score_a2_$(uname -i).tar
tar -xvf a.tar ./fused_infer_attention_score_a3_$(uname -i)

# replace fused_infer_attention_score operation files
cd $ASCEND_TOOLKIT_HOME/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b
rm -rf fused_infer_attention_score
cp -r /vllm-workspace/fused_infer_attention_score_a3_$(uname -i)/fused_infer_attention_score .

# replace related so
cd $ASCEND_TOOLKIT_HOME/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/$(uname -i)
cp /vllm-workspace/fused_infer_attention_score_a3_$(uname -i)/*.so .
