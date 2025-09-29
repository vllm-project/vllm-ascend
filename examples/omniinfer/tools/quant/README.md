### 参数说明
--input-bf16-hf-path   原始bf16权重路径  
--output-path          生成量化权重路径  
--device               设备类型，支持cpu和npu  
--model-name           hugginface权重名称，在没有元数据配置时自动根据权重名下载配置文件  
--w4                   int4量化标识, 不加该参数时为int8量化  
--pangu-mode           pangu量化标识, 开启时量化pangu 718B权重
--next                 dspk-next量化标识, 开启时量化dspk-3.2权重

### 操作步骤  
1、拷贝元数据到output路径（注：model.safetensors.index.json需使用fp8权重的对应配置，开启pangu-mode拷贝bf16权重即可） 

2、执行量化命令  

***deepseek/kimi k2/pangu 718B:***

int8量化: python quant_deepseek_kimi2.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu"  
int4量化: python quant_deepseek_kimi2.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu" --w4  

***qwen:***  

int8量化: python quant_qwen.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu"  

***gpt-oss:*** 

gpt-oss-120b int8量化: python quant_gptoss.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu" --model-type "120b"  
gpt-oss-20b int8量化: python quant_gptoss.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu" --model-type "20b"  

***C8量化:***  先拉起服务化dump数据再使用量化工具 

（1）拉起服务化时在config文件中加入c8_calib_path,对话后将knope保存至自定义的c8_calib_path

（2）python quant_deepseek_kimi2.py --input-bf16-hf-path {bf16权重路径} --output-path {量化权重路径} --device "cpu" --w4 --c8-calib-path "your_path" --kvs-safetensor-name "your_name"  
    若只想执行c8量化，可以将if args.w4后的部分注释后执行上述命令  

### 编译步骤  
进入python目录下执行： python setup.py bdist_wheel  
