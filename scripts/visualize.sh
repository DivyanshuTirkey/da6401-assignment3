#!/bin/bash

# Check if model path is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_path>"
    exit 1
fi

MODEL_PATH=$1

# Determine model type from filename
if [[ $MODEL_PATH == *"attention"* ]]; then
    MODEL_TYPE="attention"
else
    MODEL_TYPE="vanilla"
fi

# Run connectivity visualization
python visualization/connectivity_viz.py \
    --model_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --wandb_project transliteration

echo "Visualization complete!"
