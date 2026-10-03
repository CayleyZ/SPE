#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${MODEL_PATH:?Set MODEL_PATH to the downloaded ModelScope checkpoint}"
task="${1:-sst2}"
if [ "$#" -gt 0 ]; then shift; fi
case "$task" in
  cola|mrpc|rte|stsb|mr|subj) epochs=20 ;;
  qqp|mnli) epochs=5 ;;
  sst2|sst5|qnli|ag_news|imdb) epochs=10 ;;
  *) echo "Unsupported task: $task" >&2; exit 1 ;;
esac
python finetune.py --task_name "$task" --model_name_or_path "$MODEL_PATH" \
  --output_dir "${OUTPUT_DIR:-results/$task}" \
  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 --weight_decay 0 --num_warmup_steps 0 \
  --gradient_accumulation_steps 1 --num_train_epochs "$epochs" --seed 42 "$@"
