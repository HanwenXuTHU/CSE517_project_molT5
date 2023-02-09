EPOCH=30
LR=5e-4
BATCH_SIZE=16
MAX_LEN=128
OPTION=laituan245/molt5-small
DATADIR=/data/xuhw/Courses/NLP/MolT5_reimplementation/data/
SAVEDIR=/data/xuhw/Courses/NLP/MolT5_reimplementation/results/smiles2text/MolT5_small_epoch$EPOCH-lr$LR-batch$BATCH_SIZE-maxlen$MAX_LEN

python inference.py \
    --epoch $EPOCH \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --max_len $MAX_LEN \
    --option $OPTION \
    --data_path $DATADIR \
    --save_path $SAVEDIR \
    --device $0