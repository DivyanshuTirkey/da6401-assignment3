# Transliteration with Sequence-to-Sequence Models

This repository contains code for training and evaluating sequence-to-sequence models for transliteration from Latin script to Devanagari script. The project implements both vanilla Seq2Seq and attention-based Seq2Seq models.

## Project Overview

The goal of this project is to build models that can transliterate words from Latin script (English) to Devanagari script (Hindi). This is accomplished through:

1. Vanilla Seq2Seq model with RNN/GRU/LSTM cells
2. Attention-based Seq2Seq model for improved performance
3. Character-level connectivity visualization to understand model behavior

## Repository Structure

```
transliteration-seq2seq/
├── README.md
├── requirements.txt
├── config/
│   ├── vanilla_sweep.yaml
│   └── attention_sweep.yaml
├── models/
│   ├── encoder.py
│   ├── decoder.py
│   ├── attention_decoder.py
│   ├── seq2seq.py
│   └── attention_seq2seq.py
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── scripts/
│   ├── download_data.sh
│   ├── train_vanilla.sh
│   └── train_attention.sh
└── visualization/
    ├── attention_viz.py
    ├── connectivity_viz.py
    └── comparison_viz.py
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/transliteration-seq2seq.git
cd transliteration-seq2seq
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Dakshina dataset:
```bash
./scripts/download_data.sh
```

## Training

### Vanilla Seq2Seq Model
```bash
./scripts/train_vanilla.sh
```

### Attention-based Seq2Seq Model
```bash
./scripts/train_attention.sh
```

## Evaluation

To compare the performance of vanilla and attention models:
```bash
./scripts/evaluate.sh  
```

## Visualization

To visualize the connectivity between input and output characters:
```bash
./scripts/visualize.sh 
```

## Results

The attention-based model outperforms the vanilla model:

| Model | Exact Match Accuracy | Character Accuracy |
|-------|----------------------|-------------------|
| Vanilla | 40.96% | 86.09% |
| Attention | 42.56% | 75.04% |

### Key Findings

- The attention mechanism helps with correctly placing vowel diacritics and handling nasalization
- Both models struggle with longer sequences, but the attention model degrades more gracefully
- The attention model shows better context awareness for complex character mappings

## Detailed Report

For a comprehensive analysis of the models, visualizations, and findings, see our [Weights & Biases Report](https://wandb.ai/da24m005-iit-madras/seq2seq-attention-transliteration/reports/Assignment-3--VmlldzoxMjg1MTg3Mw).

## Connectivity Visualization

The repository includes interactive visualizations that show which input characters influence each output character during transliteration. This helps understand the model's internal decision-making process.

## Hyperparameter Optimization

We used Bayesian optimization through Weights & Biases sweeps to efficiently search the hyperparameter space. The optimal configurations for both models are included in the config directory.