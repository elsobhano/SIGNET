#!/usr/bin/env bash
# Stage I: pre-train one expert per large-scale corpus, from scratch.
dataset=CSL_News
output_dir=out/experts/${dataset}

deepspeed --include localhost:0,1,2,3 --master_port 29511 source/pre_training.py \
    --batch-size 16 \
    --gradient-accumulation-steps 8 \
    --epochs 20 \
    --opt AdamW \
    --lr 3e-4 \
    --quick_break 2048 \
    --output_dir ${output_dir} \
    --dataset ${dataset}
