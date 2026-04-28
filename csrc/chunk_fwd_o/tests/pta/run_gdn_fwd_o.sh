shapeBatch=1
seqlen=256
kNumHead=2
vNumHead=2
kHeadDim=128
vHeadDim=128
isVariedLen=0 #0 #1
tokenBatch=1 #1 #309
chunkSize=64
scale=0.08838834764831845
dtype="bf16"

device=2
useActualInput=0
useActualOutput=0

dataPath="/path/to/data"

echo 'Case: batch=' $batch ' seqlen=' $seqlen ' kNumHead=' $kNumHead  ' vNumHead=' $vNumHead ' kHeadDim=' $kHeadDim ' vHeadDim=' $vHeadDim ' isVariedLen=' $isVariedLen ' chunkSize=' $chunkSize ' dtype=' $dtype
python3 test_fwd_o.py $shapeBatch $seqlen $kNumHead $vNumHead $kHeadDim $vHeadDim $isVariedLen $tokenBatch $chunkSize $scale "$dtype" $useActualInput $useActualOutput $dataPath
python data_compare_o.py $dtype