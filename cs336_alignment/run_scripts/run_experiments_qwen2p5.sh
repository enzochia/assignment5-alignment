# uv run --active -m cs336_alignment.sft.run_sft \
#   --model_path models/bf16/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/ \
#   --train_data_name MATH \
#   --sft_data_name MATH \
#   --eval_data_names gsm8k MATH \
#   --prompt r1_zero \
#   --train_device cuda:0 \
#   --eval_device cuda:0 \
#   --checkpoint_dir outputs/sft/ckpt/ \
#   --keep_ckpt_until_epoch 3 \
#   --log_dir outputs/sft/logs/ \
#   --lr_scheduler cosine_with_min_lr \
#   --lr 0.00004 \
#   --batch_size 256 \
#   --gradient_accumulation_steps 32 \
#   --num_epochs 10 \
#   --gpu_memory_utilization 0.275 \
#   --do_eval \
#   --eval_batch_size 8 \
#   --wandb_project SFT-Qwen2.5-Math-1.5B-test \
#   --wandb_run_name sft_bs256_lr4e-5


# uv run --active -m cs336_alignment.sft.run_sft \
#   --model_path models/bf16/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/ \
#   --train_data_name MATH \
#   --sft_data_name MATH \
#   --eval_data_names gsm8k MATH \
#   --prompt r1_zero \
#   --train_device cuda:0 \
#   --eval_device cuda:0 \
#   --checkpoint_dir outputs/sft/ckpt/ \
#   --keep_ckpt_until_epoch -1 \
#   --log_dir outputs/sft/logs/ \
#   --lr_scheduler cosine_with_min_lr \
#   --lr 0.00002 \
#   --batch_size 256 \
#   --gradient_accumulation_steps 32 \
#   --num_epochs 10 \
#   --gpu_memory_utilization 0.275 \
#   --do_eval \
#   --eval_batch_size 8 \
#   --wandb_project SFT-Qwen2.5-Math-1.5B-test \
#   --wandb_run_name sft_bs256_lr2e-5

# uv run --active -m cs336_alignment.sft.run_sft \
#   --model_path models/bf16/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/ \
#   --train_data_name MATH \
#   --sft_data_name MATH \
#   --eval_data_names gsm8k MATH \
#   --prompt r1_zero \
#   --train_device cuda:0 \
#   --eval_device cuda:0 \
#   --checkpoint_dir outputs/sft/ckpt/ \
#   --keep_ckpt_until_epoch -1 \
#   --log_dir outputs/sft/logs/ \
#   --lr_scheduler cosine_with_min_lr \
#   --lr 0.00004 \
#   --batch_size 16 \
#   --gradient_accumulation_steps 2 \
#   --num_epochs 10 \
#   --gpu_memory_utilization 0.275 \
#   --do_eval \
#   --eval_batch_size 8 \
#   --wandb_project SFT-Qwen2.5-Math-1.5B-test \
#   --wandb_run_name sft_bs16_lr4e-5


uv run --active -m cs336_alignment.sft.run_sft \
  --model_path models/bf16/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/ \
  --train_data_name MATH \
  --sft_data_name MATH \
  --eval_data_names gsm8k MATH \
  --prompt r1_zero \
  --train_device cuda:0 \
  --eval_device cuda:0 \
  --checkpoint_dir outputs/sft/ckpt/ \
  --keep_ckpt_until_epoch -1 \
  --log_dir outputs/sft/logs/ \
  --lr_scheduler cosine_with_min_lr \
  --lr 0.00006 \
  --batch_size 256 \
  --gradient_accumulation_steps 32 \
  --num_epochs 10 \
  --gpu_memory_utilization 0.275 \
  --do_eval \
  --eval_batch_size 8 \
  --wandb_project SFT-Qwen2.5-Math-1.5B-test \
  --wandb_run_name sft_bs256_lr6e-5


# uv run --active -m cs336_alignment.grpo.run_grpo \
#   --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
#   --train_data_name MATH \
#   --eval_data_names gsm8k MATH \
#   --prompt r1_zero \
#   --train_device cuda:0 \
#   --eval_device cuda:0 \
#   --log_dir outputs/grpo/logs/ \
#   --grpo_clip_loss_type max_min \
#   --KL_beta 1e-1 \
#   --lr_scheduler cosine_with_min_lr \
#   --lr_warmup_percent 0 \
#   --lr 4e-5 \
#   --lr_scheduler_min_rate 0.6 \
#   --n_grpo_steps 24 \
#   --n_train_steps_per_rollout_batch 1 \
#   --cliprange 0.2 \
#   --train_batch_size 256 \
#   --gradient_accumulation_steps 64 \
#   --rollout_batch_size 256 \
#   --group_size 16 \
#   --loss_type grpo_clip \
#   --gpu_memory_utilization 0.28 \
#   --eval_every 4 \
#   --eval_batch_size 8 \
#   --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
#   --wandb_run_name grpo_ngrpo1_maxmin_bs256_lr4e-5_minlr0p6_rbs256_gs16_optimstep24_ckpt-sft-bs256-lr4-epoch0 \
#   --do_eval \
#   --do_eval_before_train



# uv run --active -m cs336_alignment.grpo.run_grpo \
#   --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
#   --train_data_name MATH \
#   --eval_data_names gsm8k MATH \
#   --prompt r1_zero \
#   --train_device cuda:0 \
#   --eval_device cuda:0 \
#   --log_dir outputs/grpo/logs/ \
#   --grpo_clip_loss_type clip_term_only \
#   --KL_beta 1e-1 \
#   --lr_scheduler cosine_with_min_lr \
#   --lr_warmup_percent 0 \
#   --lr 4e-5 \
#   --lr_scheduler_min_rate 0.6 \
#   --n_grpo_steps 24 \
#   --n_train_steps_per_rollout_batch 1 \
#   --cliprange 0.2 \
#   --train_batch_size 256 \
#   --gradient_accumulation_steps 64 \
#   --rollout_batch_size 256 \
#   --group_size 16 \
#   --loss_type grpo_clip \
#   --gpu_memory_utilization 0.28 \
#   --eval_every 4 \
#   --eval_batch_size 8 \
#   --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
#   --wandb_run_name grpo_ngrpo1_cliptermonly_bs256_lr4e-5_minlr0p6_rbs256_gs16_optimstep24_ckpt-sft-bs256-lr4-epoch0 \
#   --do_eval \
#   --do_eval_before_train


# uv run --active -m cs336_alignment.grpo.run_grpo \
#   --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
#   --train_data_name MATH \
#   --eval_data_names gsm8k MATH \
#   --prompt r1_zero \
#   --train_device cuda:0 \
#   --eval_device cuda:0 \
#   --log_dir outputs/grpo/logs/ \
#   --grpo_clip_loss_type kl \
#   --KL_beta 1e-1 \
#   --lr_scheduler cosine_with_min_lr \
#   --lr_warmup_percent 0 \
#   --lr 4e-5 \
#   --lr_scheduler_min_rate 0.6 \
#   --n_grpo_steps 24 \
#   --n_train_steps_per_rollout_batch 1 \
#   --cliprange 0.2 \
#   --train_batch_size 256 \
#   --gradient_accumulation_steps 64 \
#   --rollout_batch_size 256 \
#   --group_size 16 \
#   --loss_type grpo_clip \
#   --gpu_memory_utilization 0.28 \
#   --eval_every 4 \
#   --eval_batch_size 8 \
#   --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
#   --wandb_run_name grpo_ngrpo1_kl_bs256_lr4e-5_minlr0p6_rbs256_gs16_optimstep24_ckpt-sft-bs256-lr4-epoch0 \
#   --do_eval \
#   --do_eval_before_train


# uv run --active -m cs336_alignment.grpo.run_grpo \
#   --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
#   --train_data_name MATH \
#   --eval_data_names gsm8k MATH \
#   --prompt r1_zero \
#   --train_device cuda:0 \
#   --eval_device cuda:0 \
#   --log_dir outputs/grpo/logs/ \
#   --grpo_clip_loss_type kl \
#   --KL_beta 1e-1 \
#   --lr_scheduler cosine_with_min_lr \
#   --lr_warmup_percent 0 \
#   --lr 4e-5 \
#   --lr_scheduler_min_rate 0.6 \
#   --n_grpo_steps 24 \
#   --n_train_steps_per_rollout_batch 2 \
#   --cliprange 0.2 \
#   --train_batch_size 256 \
#   --gradient_accumulation_steps 64 \
#   --rollout_batch_size 256 \
#   --group_size 16 \
#   --loss_type grpo_clip \
#   --gpu_memory_utilization 0.28 \
#   --eval_every 4 \
#   --eval_batch_size 8 \
#   --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
#   --wandb_run_name grpo_ngrpo2_kl_bs256_lr4e-5_minlr0p6_rbs256_gs16_optimstep48_ckpt-sft-bs256-lr4-epoch0 \
#   --do_eval \
#   --do_eval_before_train

# uv run --active -m cs336_alignment.grpo.run_grpo \
#   --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
#   --train_data_name MATH \
#   --eval_data_names gsm8k MATH \
#   --prompt r1_zero \
#   --train_device cuda:0 \
#   --eval_device cuda:0 \
#   --log_dir outputs/grpo/logs/ \
#   --grpo_clip_loss_type no_kl \
#   --KL_beta 1e-1 \
#   --lr_scheduler cosine_with_min_lr \
#   --lr_warmup_percent 0 \
#   --lr 4e-5 \
#   --lr_scheduler_min_rate 0.6 \
#   --n_grpo_steps 24 \
#   --n_train_steps_per_rollout_batch 2 \
#   --cliprange 0.2 \
#   --train_batch_size 256 \
#   --gradient_accumulation_steps 64 \
#   --rollout_batch_size 256 \
#   --group_size 16 \
#   --loss_type grpo_clip \
#   --gpu_memory_utilization 0.28 \
#   --eval_every 4 \
#   --eval_batch_size 8 \
#   --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
#   --wandb_run_name grpo_ngrpo2_nokl_bs256_lr4e-5_minlr0p6_rbs256_gs16_optimstep48_ckpt-sft-bs256-lr4-epoch0 \
#   --do_eval \
#   --do_eval_before_train


  # uv run --active -m cs336_alignment.grpo.run_grpo \
  # --model_path models/bf16/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2/ \
  # --train_data_name MATH \
  # --eval_data_names gsm8k MATH \
  # --prompt r1_zero \
  # --train_device cuda:0 \
  # --eval_device cuda:0 \
  # --log_dir outputs/grpo/logs/ \
  # --grpo_clip_loss_type kl \
  # --KL_beta 1e-1 \
  # --lr_scheduler cosine_with_min_lr \
  # --lr_warmup_percent 0 \
  # --lr 4e-5 \
  # --lr_scheduler_min_rate 0.6 \
  # --n_grpo_steps 24 \
  # --n_train_steps_per_rollout_batch 1 \
  # --cliprange 0.2 \
  # --train_batch_size 256 \
  # --gradient_accumulation_steps 64 \
  # --rollout_batch_size 256 \
  # --group_size 16 \
  # --loss_type grpo_clip \
  # --gpu_memory_utilization 0.28 \
  # --eval_every 4 \
  # --eval_batch_size 8 \
  # --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
  # --wandb_run_name grpo_ngrpo1_kl_bs256_lr4e-5_minlr0p6_rbs256_gs16_optimstep24_ckpt-orig-qwen \
  # --do_eval \
  # --do_eval_before_train


# uv run --active -m cs336_alignment.grpo.run_grpo \
#   --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
#   --train_data_name MATH \
#   --eval_data_names gsm8k MATH \
#   --prompt r1_zero \
#   --train_device cuda:0 \
#   --eval_device cuda:0 \
#   --log_dir outputs/grpo/logs/ \
#   --grpo_clip_loss_type max_min \
#   --KL_beta 1e-1 \
#   --lr_scheduler cosine_with_min_lr \
#   --lr_warmup_percent 0 \
#   --lr 4e-5 \
#   --lr_scheduler_min_rate 0.6 \
#   --n_grpo_steps 24 \
#   --n_train_steps_per_rollout_batch 1 \
#   --cliprange 0.2 \
#   --train_batch_size 384 \
#   --gradient_accumulation_steps 96 \
#   --rollout_batch_size 384 \
#   --group_size 16 \
#   --loss_type grpo_clip \
#   --gpu_memory_utilization 0.28 \
#   --eval_every 4 \
#   --eval_batch_size 8 \
#   --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
#   --wandb_run_name grpo_ngrpo1_maxmin_bs384_lr4e-5_minlr0p6_rbs384_gs16_optimstep24_ckpt-sft-bs256-lr4-epoch0 \
#   --do_eval \
#   --do_eval_before_train


# uv run --active -m cs336_alignment.grpo.run_grpo \
#   --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
#   --train_data_name MATH \
#   --eval_data_names gsm8k MATH \
#   --prompt r1_zero \
#   --train_device cuda:0 \
#   --eval_device cuda:0 \
#   --log_dir outputs/grpo/logs/ \
#   --grpo_clip_loss_type clip_term_only \
#   --KL_beta 1e-1 \
#   --lr_scheduler cosine_with_min_lr \
#   --lr_warmup_percent 0 \
#   --lr 4e-5 \
#   --lr_scheduler_min_rate 0.6 \
#   --n_grpo_steps 24 \
#   --n_train_steps_per_rollout_batch 1 \
#   --cliprange 0.2 \
#   --train_batch_size 384 \
#   --gradient_accumulation_steps 96 \
#   --rollout_batch_size 384 \
#   --group_size 16 \
#   --loss_type grpo_clip \
#   --gpu_memory_utilization 0.28 \
#   --eval_every 4 \
#   --eval_batch_size 8 \
#   --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
#   --wandb_run_name grpo_ngrpo1_cliptermonly_bs384_lr4e-5_minlr0p6_rbs384_gs16_optimstep24_ckpt-sft-bs256-lr4-epoch0 \
#   --do_eval \
#   --do_eval_before_train


# uv run --active -m cs336_alignment.grpo.run_grpo \
#   --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
#   --train_data_name MATH \
#   --eval_data_names gsm8k MATH \
#   --prompt r1_zero \
#   --train_device cuda:0 \
#   --eval_device cuda:0 \
#   --log_dir outputs/grpo/logs/ \
#   --grpo_clip_loss_type kl \
#   --KL_beta 1e-1 \
#   --lr_scheduler cosine_with_min_lr \
#   --lr_warmup_percent 0 \
#   --lr 4e-5 \
#   --lr_scheduler_min_rate 0.6 \
#   --n_grpo_steps 24 \
#   --n_train_steps_per_rollout_batch 1 \
#   --cliprange 0.2 \
#   --train_batch_size 384 \
#   --gradient_accumulation_steps 96 \
#   --rollout_batch_size 384 \
#   --group_size 16 \
#   --loss_type grpo_clip \
#   --gpu_memory_utilization 0.28 \
#   --eval_every 4 \
#   --eval_batch_size 8 \
#   --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
#   --wandb_run_name grpo_ngrpo1_kl_bs384_lr4e-5_minlr0p6_rbs384_gs16_optimstep24_ckpt-sft-bs256-lr4-epoch0 \
#   --do_eval \
#   --do_eval_before_train


# uv run --active -m cs336_alignment.grpo.run_grpo \
#   --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
#   --train_data_name MATH \
#   --eval_data_names gsm8k MATH \
#   --prompt r1_zero \
#   --train_device cuda:0 \
#   --eval_device cuda:0 \
#   --log_dir outputs/grpo/logs/ \
#   --grpo_clip_loss_type kl \
#   --KL_beta 1e-1 \
#   --lr_scheduler cosine_with_min_lr \
#   --lr_warmup_percent 0 \
#   --lr 4e-5 \
#   --lr_scheduler_min_rate 0.6 \
#   --n_grpo_steps 24 \
#   --n_train_steps_per_rollout_batch 1 \
#   --cliprange 0.2 \
#   --train_batch_size 128 \
#   --gradient_accumulation_steps 32 \
#   --rollout_batch_size 128 \
#   --group_size 16 \
#   --loss_type grpo_clip \
#   --gpu_memory_utilization 0.28 \
#   --eval_every 4 \
#   --eval_batch_size 8 \
#   --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
#   --wandb_run_name grpo_ngrpo1_kl_bs128_lr4e-5_minlr0p6_rbs128_gs16_optimstep24_ckpt-sft-bs256-lr4-epoch0 \
#   --do_eval \
#   --do_eval_before_train


# uv run --active -m cs336_alignment.grpo.run_grpo \
#   --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
#   --train_data_name MATH \
#   --eval_data_names gsm8k MATH \
#   --prompt r1_zero \
#   --train_device cuda:0 \
#   --eval_device cuda:0 \
#   --log_dir outputs/grpo/logs/ \
#   --grpo_clip_loss_type clip_term_only \
#   --KL_beta 1e-1 \
#   --lr_scheduler cosine_with_min_lr \
#   --lr_warmup_percent 0 \
#   --lr 4e-5 \
#   --lr_scheduler_min_rate 0.6 \
#   --n_grpo_steps 24 \
#   --n_train_steps_per_rollout_batch 1 \
#   --cliprange 0.2 \
#   --train_batch_size 128 \
#   --gradient_accumulation_steps 32 \
#   --rollout_batch_size 128 \
#   --group_size 16 \
#   --loss_type grpo_clip \
#   --gpu_memory_utilization 0.28 \
#   --eval_every 4 \
#   --eval_batch_size 8 \
#   --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
#   --wandb_run_name grpo_ngrpo1_cliptermonly_bs128_lr4e-5_minlr0p6_rbs128_gs16_optimstep24_ckpt-sft-bs256-lr4-epoch0 \
#   --do_eval \
#   --do_eval_before_train


# uv run --active -m cs336_alignment.grpo.run_grpo \
#   --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
#   --train_data_name MATH \
#   --eval_data_names gsm8k MATH \
#   --prompt r1_zero \
#   --train_device cuda:0 \
#   --eval_device cuda:0 \
#   --log_dir outputs/grpo/logs/ \
#   --grpo_clip_loss_type max_min \
#   --KL_beta 1e-1 \
#   --lr_scheduler cosine_with_min_lr \
#   --lr_warmup_percent 0 \
#   --lr 4e-5 \
#   --lr_scheduler_min_rate 0.6 \
#   --n_grpo_steps 24 \
#   --n_train_steps_per_rollout_batch 1 \
#   --cliprange 0.2 \
#   --train_batch_size 128 \
#   --gradient_accumulation_steps 32 \
#   --rollout_batch_size 128 \
#   --group_size 16 \
#   --loss_type grpo_clip \
#   --gpu_memory_utilization 0.28 \
#   --eval_every 4 \
#   --eval_batch_size 8 \
#   --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
#   --wandb_run_name grpo_ngrpo1_maxmin_bs128_lr4e-5_minlr0p6_rbs128_gs16_optimstep24_ckpt-sft-bs256-lr4-epoch0 \
#   --do_eval \
#   --do_eval_before_train


uv run --active -m cs336_alignment.grpo.run_grpo \
  --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
  --train_data_name MATH \
  --eval_data_names gsm8k MATH \
  --prompt r1_zero \
  --train_device cuda:0 \
  --eval_device cuda:0 \
  --log_dir outputs/grpo/logs/ \
  --grpo_clip_loss_type max_min \
  --KL_beta 1e-1 \
  --lr_scheduler cosine_with_min_lr \
  --lr_warmup_percent 0 \
  --lr 4e-5 \
  --lr_scheduler_min_rate 0.6 \
  --n_grpo_steps 48 \
  --n_train_steps_per_rollout_batch 1 \
  --cliprange 0.2 \
  --train_batch_size 512 \
  --gradient_accumulation_steps 128 \
  --rollout_batch_size 512 \
  --group_size 16 \
  --loss_type grpo_clip \
  --gpu_memory_utilization 0.28 \
  --eval_every 4 \
  --eval_batch_size 8 \
  --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
  --wandb_run_name grpo_ngrpo1_maxmin_bs512_lr4e-5_minlr0p6_rbs512_gs16_optimstep48_ckpt-sft-bs256-lr4-epoch0 \
  --do_eval \
  --do_eval_before_train

uv run --active -m cs336_alignment.grpo.run_grpo \
  --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
  --train_data_name MATH \
  --eval_data_names gsm8k MATH \
  --prompt r1_zero \
  --train_device cuda:0 \
  --eval_device cuda:0 \
  --log_dir outputs/grpo/logs/ \
  --grpo_clip_loss_type clip_term_only \
  --KL_beta 1e-1 \
  --lr_scheduler cosine_with_min_lr \
  --lr_warmup_percent 0 \
  --lr 4e-5 \
  --lr_scheduler_min_rate 0.6 \
  --n_grpo_steps 48 \
  --n_train_steps_per_rollout_batch 1 \
  --cliprange 0.2 \
  --train_batch_size 512 \
  --gradient_accumulation_steps 128 \
  --rollout_batch_size 512 \
  --group_size 16 \
  --loss_type grpo_clip \
  --gpu_memory_utilization 0.28 \
  --eval_every 4 \
  --eval_batch_size 8 \
  --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
  --wandb_run_name grpo_ngrpo1_cliptermonly_bs512_lr4e-5_minlr0p6_rbs512_gs16_optimstep48_ckpt-sft-bs256-lr4-epoch0 \
  --do_eval \
  --do_eval_before_train


uv run --active -m cs336_alignment.grpo.run_grpo \
  --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
  --train_data_name MATH \
  --eval_data_names gsm8k MATH \
  --prompt r1_zero \
  --train_device cuda:0 \
  --eval_device cuda:0 \
  --log_dir outputs/grpo/logs/ \
  --grpo_clip_loss_type kl \
  --KL_beta 1e-1 \
  --lr_scheduler cosine_with_min_lr \
  --lr_warmup_percent 0 \
  --lr 4e-5 \
  --lr_scheduler_min_rate 0.6 \
  --n_grpo_steps 48 \
  --n_train_steps_per_rollout_batch 1 \
  --cliprange 0.2 \
  --train_batch_size 512 \
  --gradient_accumulation_steps 128 \
  --rollout_batch_size 512 \
  --group_size 16 \
  --loss_type grpo_clip \
  --gpu_memory_utilization 0.28 \
  --eval_every 4 \
  --eval_batch_size 8 \
  --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
  --wandb_run_name grpo_ngrpo1_kl_bs512_lr4e-5_minlr0p6_rbs512_gs16_optimstep48_ckpt-sft-bs256-lr4-epoch0 \
  --do_eval \
  --do_eval_before_train



uv run --active -m cs336_alignment.grpo.run_grpo \
  --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
  --train_data_name MATH \
  --eval_data_names gsm8k MATH \
  --prompt r1_zero \
  --train_device cuda:0 \
  --eval_device cuda:0 \
  --log_dir outputs/grpo/logs/ \
  --grpo_clip_loss_type max_min \
  --KL_beta 1e-1 \
  --lr_scheduler cosine_with_min_lr \
  --lr_warmup_percent 0 \
  --lr 4e-5 \
  --lr_scheduler_min_rate 0.6 \
  --n_grpo_steps 48 \
  --n_train_steps_per_rollout_batch 1 \
  --cliprange 0.2 \
  --train_batch_size 64 \
  --gradient_accumulation_steps 16 \
  --rollout_batch_size 64 \
  --group_size 16 \
  --loss_type grpo_clip \
  --gpu_memory_utilization 0.28 \
  --eval_every 4 \
  --eval_batch_size 8 \
  --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
  --wandb_run_name grpo_ngrpo1_maxmin_bs64_lr4e-5_minlr0p6_rbs64_gs16_optimstep48_ckpt-sft-bs256-lr4-epoch0 \
  --do_eval \
  --do_eval_before_train

uv run --active -m cs336_alignment.grpo.run_grpo \
  --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
  --train_data_name MATH \
  --eval_data_names gsm8k MATH \
  --prompt r1_zero \
  --train_device cuda:0 \
  --eval_device cuda:0 \
  --log_dir outputs/grpo/logs/ \
  --grpo_clip_loss_type clip_term_only \
  --KL_beta 1e-1 \
  --lr_scheduler cosine_with_min_lr \
  --lr_warmup_percent 0 \
  --lr 4e-5 \
  --lr_scheduler_min_rate 0.6 \
  --n_grpo_steps 48 \
  --n_train_steps_per_rollout_batch 1 \
  --cliprange 0.2 \
  --train_batch_size 64 \
  --gradient_accumulation_steps 16 \
  --rollout_batch_size 64 \
  --group_size 16 \
  --loss_type grpo_clip \
  --gpu_memory_utilization 0.28 \
  --eval_every 4 \
  --eval_batch_size 8 \
  --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
  --wandb_run_name grpo_ngrpo1_cliptermonly_bs64_lr4e-5_minlr0p6_rbs64_gs16_optimstep48_ckpt-sft-bs256-lr4-epoch0 \
  --do_eval \
  --do_eval_before_train


uv run --active -m cs336_alignment.grpo.run_grpo \
  --model_path outputs/sft/ckpt/sft_bs256_lr4e-5/ckpt_epoch_0_6steps/ \
  --train_data_name MATH \
  --eval_data_names gsm8k MATH \
  --prompt r1_zero \
  --train_device cuda:0 \
  --eval_device cuda:0 \
  --log_dir outputs/grpo/logs/ \
  --grpo_clip_loss_type kl \
  --KL_beta 1e-1 \
  --lr_scheduler cosine_with_min_lr \
  --lr_warmup_percent 0 \
  --lr 4e-5 \
  --lr_scheduler_min_rate 0.6 \
  --n_grpo_steps 48 \
  --n_train_steps_per_rollout_batch 1 \
  --cliprange 0.2 \
  --train_batch_size 64 \
  --gradient_accumulation_steps 16 \
  --rollout_batch_size 64 \
  --group_size 16 \
  --loss_type grpo_clip \
  --gpu_memory_utilization 0.28 \
  --eval_every 4 \
  --eval_batch_size 8 \
  --wandb_project GRPO-Qwen2.5-Math-1.5B-test \
  --wandb_run_name grpo_ngrpo1_kl_bs64_lr4e-5_minlr0p6_rbs64_gs16_optimstep48_ckpt-sft-bs256-lr4-epoch0 \
  --do_eval \
  --do_eval_before_train