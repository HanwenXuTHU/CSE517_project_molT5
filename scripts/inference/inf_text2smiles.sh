EPOCH=30
LR=5e-4
BATCH_SIZE=16
MAX_LEN=128
OPTION=laituan245/molt5-small
DATADIR=./data/
SAVEDIR=./results/text2smiles/MolT5_small_epoch$EPOCH-lr$LR-batch$BATCH_SIZE-maxlen$MAX_LEN
DEVICE=$1

python inference_text2smiles.py \
    --epoch $EPOCH \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --max_len $MAX_LEN \
    --option $OPTION \
    --data_path $DATADIR \
    --save_path $SAVEDIR \
    --device $DEVICE