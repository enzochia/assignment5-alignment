uv run --active -m cs336_alignment.math_baseline.math_baseline \
  --model models/ii/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2 \
  --data MATH \
  --data_path data/MATH/validation.jsonl \
  --reward r1_zero \
  --save_to outputs/math_baseline/
