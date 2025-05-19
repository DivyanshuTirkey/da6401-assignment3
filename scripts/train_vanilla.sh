#!/bin/bash

# Download data if not exists
if [ ! -d "dakshina_dataset_v1.0" ]; then
    ./scripts/download_data.sh
fi

# Train vanilla model
python src/train.py \
    --config config/vanilla_sweep.yaml \
    --model_type vanilla \
    --wandb_project transliteration \
    --max_epochs 30 \
    --patience 5
