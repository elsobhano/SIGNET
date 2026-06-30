#!/usr/bin/env bash
# Stage II: contrastive alignment of the gated fusion network.
dataset=Phoenix
task=contrastive_loss
k=2
output_dir=out/

# experts from Stage I (must match Stage III)
experts="out/experts/CSL_News/best_checkpoint.pth \
         out/experts/YT_ASL/best_checkpoint.pth \
         out/experts/BOBSL/best_checkpoint.pth"

ckpt_path=./pretrained_weight/pretrained.pth

deepspeed --include localhost:0 --master_port 29511 source/gating_contrastive.py \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --epochs 20 \
    --opt AdamW \
    --lr 8e-4 \
    --router_lr 1e-3 \
    --warmup-epochs 1 \
    --output_dir ${output_dir} \
    --finetune ${ckpt_path} \
    --expert_model_paths ${experts} \
    --K ${k} \
    --dataset ${dataset} \
    --task ${task}
