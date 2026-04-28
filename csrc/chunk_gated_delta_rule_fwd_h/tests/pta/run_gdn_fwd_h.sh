batch=1 #309
seqlen=256
kNumHead=2
vNumHead=2
kHeadDim=128
vHeadDim=128
isVariedLen=0 #1
chunkSize=64
dtype="bf16"
useActualInput=0
useActualOutput=0
device=2
useInitialState=0
storeFinalState=0
dataPath="/path/to/data"

echo 'Case: batch=' $batch ' seqlen=' $seqlen ' kNumHead=' $kNumHead  ' vNumHead=' $vNumHead ' kHeadDim=' $kHeadDim ' vHeadDim=' $vHeadDim ' isVariedLen=' $isVariedLen ' chunkSize=' $chunkSize ' dtype=' $dtype
python3 test_fwd_h.py $batch $seqlen $kNumHead $vNumHead $kHeadDim $vHeadDim $isVariedLen $chunkSize $useInitialState $storeFinalState "$dtype" $useActualInput $useActualOutput $dataPath
python data_compare_h.py $dtype