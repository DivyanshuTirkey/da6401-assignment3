#!/bin/bash

# Download data if not exists
if [ ! -d "dakshina_dataset_v1.0" ]; then
    ./scripts/download_data.sh
fi

# Train attention model
python src/train.py \
    --config config/attention_sweep.yaml \
    --model_type attention \
    --wandb_project transliteration \
    --max_epochs 30 \
    --patience 5
