#!/usr/bin/env bash

DATA_DIR=$1
RESULT_DIR=$2

python train.py \
  $DATA_DIR \
  --task language_modeling \
  --save-dir $RESULT_DIR \
  --arch transformer_lm --share-decoder-input-output-embed \
  --dropout 0.1 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --tokens-per-sample 512 --sample-break-mode none \
  --max-tokens 1024 --update-freq 8 \
  --fp16 \
  --max-update 50000
