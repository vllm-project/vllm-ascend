# TODO
load ssd or d2d transformer for expert weight

matrixaccLib-EPLB:

Input 热度表

output 
加载到hbm的 tensor


step1. collect

step2. eplb algo
step3. expert weight loading(ssd->host->hbm or d2d hbm) hbm buffer,  与后处理或者attention 计算掩盖

step4. expert table apply & hbm buffer copy



