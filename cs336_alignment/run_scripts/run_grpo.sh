uv run --active -m cs336_alignment.grpo.run_grpo \
  --model_path outputs/ckpt/sft_dev_run_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
  --data_train_path data/MATH/train.jsonl \
  --data_eval_path data/MATH/validation.jsonl \
  --prompt r1_zero \
  --train_device cuda:0 \
  --eval_device cuda:0 \
  --log_dir outputs/logs/grpo/ \
  --grpo_clip_loss_type max_min \
  --KL_beta 5e-1 \
  --lr_scheduler cosine_with_min_lr \
  --lr_warmup_percent 0 \
  --lr 0.00004 \
  --lr_scheduler_min_rate 0.6 \
  --n_grpo_steps 24 \
  --n_train_steps_per_rollout_batch 2 \
  --cliprange 0.2 \
  --train_batch_size 256 \
  --gradient_accumulation_steps 64 \
  --rollout_batch_size 256 \
  --group_size 16 \
  --loss_type grpo_clip \
  --gpu_memory_utilization 0.28 \
  --eval_every 4 \
  --eval_batch_size 8 \
  --wandb_project GRPO-Qwen2.5-Math-1.5B-dev \
  --wandb_run_name grpo_dev_run_maxmin_bs256_lr4e-5_minlr0p6_ngrpo2_rbs256_gs16_optimstep24_ckpt-bs256-lr4-epoch0 \
  --do_eval \
  --do_eval_before_train \
  # Note: n_grpo_steps changes along with rollout_batch_size
  # --model_path outputs/ckpt/sft_dev_run_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
  # --model_path models/bf16/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/ \
  # --checkpoint_dir outputs/ckpt/grpo/ \
  # --eval_size 1024 \
  # --normalize_by_std