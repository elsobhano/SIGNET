#!/usr/bin/env bash
# Stage III: SLT fine-tuning of the gated mixture-of-experts.
dataset=Phoenix
task=SLT
output_dir=out/${dataset}/signet

# experts from Stage I (must match Stage II)
experts="out/experts/CSL_News/best_checkpoint.pth \
         out/experts/YT_ASL/best_checkpoint.pth \
         out/experts/BOBSL/best_checkpoint.pth"

# gate checkpoint from Stage II
ckpt_path=out/Phoenix/gating-contrastive/<run>/best_checkpoint_loss.pth

deepspeed --include localhost:0 --master_port 29511 source/gating.py \
    --batch-size 8 \
    --gradient-accumulation-steps 1 \
    --epochs 40 \
    --opt AdamW \
    --lr 1e-4 \
    --router_lr 3e-4 \
    --warmup-epochs 2 \
    --output_dir ${output_dir} \
    --finetune ${ckpt_path} \
    --expert_model_paths ${experts} \
    --dataset ${dataset} \
    --task ${task}
