#!/usr/bin/env bash
# Evaluate a trained SIGNET model on a single GPU.
dataset=Phoenix
task=SLT
ckpt_path=out/${dataset}/signet/best_checkpoint.pth
output_dir=eval/${dataset}

# experts must match training
experts="out/experts/CSL_News/best_checkpoint.pth \
         out/experts/YT_ASL/best_checkpoint.pth \
         out/experts/BOBSL/best_checkpoint.pth"

deepspeed --include localhost:0 --master_port 29511 source/gating.py \
    --batch-size 8 \
    --gradient-accumulation-steps 1 \
    --output_dir ${output_dir} \
    --finetune ${ckpt_path} \
    --expert_model_paths ${experts} \
    --K 2 \
    --dataset ${dataset} \
    --task ${task} \
    --eval
