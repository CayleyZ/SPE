#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# TOKENIZED_DATASET must contain train and validation splits saved by datasets.
: "${TOKENIZED_DATASET:?Set TOKENIZED_DATASET to your preprocessed corpus directory}"
torchrun --standalone --nproc_per_node="${NUM_GPUS:-8}" pretrain.py \
  --tokenized_dataset "$TOKENIZED_DATASET" \
  --config_name configs/spe_110m.json \
  --tokenizer_name "${TOKENIZER_PATH:-bert-base-uncased}" \
  --output_dir "${OUTPUT_DIR:-checkpoints/spe_110m}" \
  --per_device_train_batch_size 64 --per_device_eval_batch_size 64 \
  --max_seq_length 128 --max_train_steps 650000 \
  --learning_rate 2e-4 --weight_decay 0 --num_warmup_steps 5000 \
  --mlm_probability 0.15 --gradient_accumulation_steps 1 \
  --checkpointing_steps 50000 --seed 42 "$@"
