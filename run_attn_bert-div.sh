#!/usr/bin/env bash


DATA_DIR=$1
RESULT_DIR=$2

echo '********** Bert Training **********'

python train.py \
	$DATA_DIR --fp16 --num-workers 8 --ddp-backend=c10d \
  --task masked_lm --criterion masked_lm \
  --arch roberta_base --sample-break-mode complete_doc --shorten-method random_crop --tokens-per-sample 512 \
  --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-6 --clip-norm 1.0 \
  --lr-scheduler polynomial_decay --lr 0.0005 --warmup-updates 10000 --total-num-update 1000000 \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --max-tokens 8192 --update-freq 2 --seed 100 \
  --mask-prob 0.15 \
  --keep-updates-list 10000 50000 100000 200000 400000 600000 800000 1000000 \
  --max-update 1000000 --log-format simple --log-interval 10 --tensorboard-logdir . \
  --save-interval-updates 1000 --keep-interval-updates 15 --no-epoch-checkpoints --skip-invalid-size-inputs-valid-test \
  --save-dir $RESULT_DIR