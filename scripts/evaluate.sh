#!/bin/bash

# Check if model paths are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <vanilla_model_path> <attention_model_path>"
    exit 1
fi

VANILLA_MODEL_PATH=$1
ATTENTION_MODEL_PATH=$2

# Run comparison
python visualization/comparison_viz.py \
    --vanilla_model_path $VANILLA_MODEL_PATH \
    --attention_model_path $ATTENTION_MODEL_PATH \
    --wandb_project transliteration \
    --run_name "model_comparison"

echo "Evaluation complete!"
