uv run --active -m cs336_alignment.grpo.run_grpo \
  --model_path outputs/ckpt/ckpt_2epoch_220steps/ \
  --data_train_path data/MATH/train.jsonl \
  --data_eval_path data/MATH/validation.jsonl \
  --prompt r1_zero \
  --train_device cuda:0 \
  --eval_device cuda:0 \
  --checkpoint_dir outputs/ckpt/grpo/ \
  --log_dir outputs/logs/grpo/ \
  --lr_scheduler cosine_with_min_lr \
  --lr 0.00004 \
  --n_grpo_steps 200 \
  --n_train_steps_per_rollout_batch 2 \
  --cliprange 0.2 \
  --train_batch_size 8 \
  --gradient_accumulation_steps 8 \
  --rollout_batch_size 4 \
  --group_size 4 \
  --loss_type grpo_clip \
  --gpu_memory_utilization 0.28 \
  --eval_every 10 \
  --eval_batch_size 2 \
  --do_eval \
  # --do_eval_before_train \
  # --normalize_by_std