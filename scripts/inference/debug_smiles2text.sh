EPOCH=30
LR=5e-4
BATCH_SIZE=16
MAX_LEN=128
OPTION=laituan245/molt5-small
DATADIR=./data/
SAVEDIR=./results/smiles2text/MolT5_small_epoch$EPOCH-lr$LR-batch$BATCH_SIZE-maxlen$MAX_LEN
DEVICE=3

python inference.py \
    --epoch $EPOCH \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --max_len $MAX_LEN \
    --option $OPTION \
    --data_path $DATADIR \
    --save_path $SAVEDIR \
    --device $DEVICE \
    --debug