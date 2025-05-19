import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import wandb

from src.utils import create_vocab, text_to_indices, plot_attention, create_attention_grid
from src.evaluate import translate_sentence

def visualize_attention(model, src_text, src_vocab, tgt_vocab, idx2tgt, idx2src):
    """Visualize attention weights for a single example"""
    translation, attention_weights = translate_sentence(model, src_text, src_vocab, tgt_vocab, idx2tgt)
    
    if not attention_weights:
        print(f"No attention weights available for model. Translation: {translation}")
        return None
    
    fig = plot_attention(src_text, translation, np.array(attention_weights), idx2src)
    return fig, translation

def create_attention_visualizations(model, test_examples, src_vocab, tgt_vocab, idx2tgt, idx2src):
    """Create attention visualizations for multiple examples"""
    results = []
    figures = []
    
    for example in test_examples:
        fig, translation = visualize_attention(model, example, src_vocab, tgt_vocab, idx2tgt, idx2src)
        if fig:
            figures.append(fig)
        results.append(f"Input: {example}, Translation: {translation}")
    
    # Create grid of attention plots
    grid_fig = create_attention_grid(model, test_examples, src_vocab, tgt_vocab, idx2src, idx2tgt)
    
    return figures, grid_fig, results

def log_to_wandb(figures, grid_fig, results, run_name=None):
    """Log attention visualizations to W&B"""
    if run_name:
        wandb.init(project="transliteration", name=run_name)
    else:
        wandb.init(project="transliteration")
    
    # Log individual attention plots
    for i, fig in enumerate(figures):
        wandb.log({f"attention_{test_examples[i]}": wandb.Image(fig)})
        plt.close(fig)
    
    # Log grid of attention plots
    wandb.log({"attention_grid": wandb.Image(grid_fig)})
    plt.close(grid_fig)
    
    # Log examples as a table
    wandb.log({"examples": wandb.Table(columns=["Example"], data=[[r] for r in results])})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize attention weights')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/', 
                        help='Path to data directory')
    parser.add_argument('--wandb_project', type=str, default='transliteration', 
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None, 
                        help='W&B run name')
    args = parser.parse_args()
    
    # Load data and create vocabularies
    train_path = f"{args.data_path}/hi.translit.sampled.train.tsv"
    train_df = pd.read_csv(train_path, delimiter='\t', names=['hi', 'en', '_'])
    
    src_vocab = create_vocab(train_df['en'])
    tgt_vocab = create_vocab(train_df['hi'])
    
    # Create reverse mappings
    idx2src = {idx: char for char, idx in src_vocab.items()}
    idx2tgt = {idx: char for char, idx in tgt_vocab.items()}
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model config
    config = {
        'input_vocab_size': len(src_vocab),
        'output_vocab_size': len(tgt_vocab),
        'embedding_dim': 256,  # Default values, will be overridden by loaded weights
        'hidden_dim': 512,
        'num_encoding_layers': 2,
        'num_decoding_layers': 2,
        'dropout': 0.3,
        'cell_type': 'gru',
        'device': device
    }
    
    # Initialize model
    from models.attention_seq2seq import AttentionSeq2Seq
    model = AttentionSeq2Seq(config).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Test examples
    test_examples = ["ankur", "uncle", "ankganit", "hindi", "bharat", "computer", "india", "delhi", "mumbai"]
    
    # Create visualizations
    figures, grid_fig, results = create_attention_visualizations(
        model, test_examples, src_vocab, tgt_vocab, idx2tgt, idx2src
    )
    
    # Log to W&B
    log_to_wandb(figures, grid_fig, results, args.run_name)
