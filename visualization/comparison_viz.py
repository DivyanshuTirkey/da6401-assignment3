import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import wandb

from src.utils import create_vocab, text_to_indices
from src.evaluate import translate_sentence, generate_predictions

def compare_models(vanilla_model, attention_model, test_loader, src_vocab, tgt_vocab, idx2src, idx2tgt):
    """Compare vanilla and attention models"""
    # Generate predictions
    vanilla_preds = generate_predictions(vanilla_model, test_loader, src_vocab, tgt_vocab, idx2tgt)
    attention_preds = generate_predictions(attention_model, test_loader, src_vocab, tgt_vocab, idx2tgt)
    
    # Calculate overall accuracy
    vanilla_acc = vanilla_preds['Correct'].mean()
    attention_acc = attention_preds['Correct'].mean()
    
    print(f"Vanilla Model Accuracy: {vanilla_acc:.4f}")
    print(f"Attention Model Accuracy: {attention_acc:.4f}")
    print(f"Improvement: {(attention_acc - vanilla_acc):.4f} ({(attention_acc/vanilla_acc - 1)*100:.2f}%)")
    
    # Merge predictions
    merged_df = pd.merge(
        vanilla_preds, attention_preds,
        on=['Source', 'Target'],
        suffixes=('_vanilla', '_attention')
    )
    
    # Find examples where models disagree
    vanilla_only_correct = merged_df[
        (merged_df['Correct_vanilla'] == True) & 
        (merged_df['Correct_attention'] == False)
    ]
    
    attention_only_correct = merged_df[
        (merged_df['Correct_vanilla'] == False) & 
        (merged_df['Correct_attention'] == True)
    ]
    
    print(f"\nExamples where only vanilla model is correct: {len(vanilla_only_correct)}")
    print(f"Examples where only attention model is correct: {len(attention_only_correct)}")
    
    # Analyze by sequence length
    merged_df['source_length'] = merged_df['Source'].apply(len)
    
    # Group by length and calculate accuracy
    length_comparison = merged_df.groupby('source_length').agg({
        'Correct_vanilla': 'mean',
        'Correct_attention': 'mean',
        'Source': 'count'
    }).rename(columns={'Source': 'count'}).reset_index()
    
    return {
        'vanilla_acc': vanilla_acc,
        'attention_acc': attention_acc,
        'vanilla_preds': vanilla_preds,
        'attention_preds': attention_preds,
        'merged_df': merged_df,
        'vanilla_only_correct': vanilla_only_correct,
        'attention_only_correct': attention_only_correct,
        'length_comparison': length_comparison
    }

def plot_comparison_results(results):
    """Plot comparison results"""
    # Accuracy by sequence length
    length_comp = results['length_comparison']
    
    plt.figure(figsize=(12, 6))
    plt.plot(length_comp['source_length'], length_comp['Correct_vanilla'], 'o-', label='Vanilla Model')
    plt.plot(length_comp['source_length'], length_comp['Correct_attention'], 'o-', label='Attention Model')
    plt.xlabel('Source Sequence Length')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy vs Sequence Length')
    plt.legend()
    plt.grid(True)
    
    # Add sample counts
    for i, row in length_comp.iterrows():
        plt.annotate(f"n={row['count']}", 
                     (row['source_length'], row['Correct_vanilla']),
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
    
    length_fig = plt.gcf()
    plt.close()
    
    # Error analysis
    merged_df = results['merged_df']
    
    # Add error type column
    merged_df['error_type'] = 'Both Correct'
    merged_df.loc[(merged_df['Correct_vanilla'] == False) & (merged_df['Correct_attention'] == False), 'error_type'] = 'Both Wrong'
    merged_df.loc[(merged_df['Correct_vanilla'] == True) & (merged_df['Correct_attention'] == False), 'error_type'] = 'Only Vanilla Correct'
    merged_df.loc[(merged_df['Correct_vanilla'] == False) & (merged_df['Correct_attention'] == True), 'error_type'] = 'Only Attention Correct'
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    error_counts = merged_df['error_type'].value_counts()
    plt.pie(error_counts, labels=error_counts.index, autopct='%1.1f%%')
    plt.title('Error Distribution Between Models')
    
    error_fig = plt.gcf()
    plt.close()
    
    return {
        'length_fig': length_fig,
        'error_fig': error_fig
    }

def log_comparison_to_wandb(results, figures, run_name=None):
    """Log comparison results to W&B"""
    if run_name:
        wandb.init(project="transliteration", name=run_name)
    else:
        wandb.init(project="transliteration")
    
    # Log metrics
    wandb.log({
        'vanilla_accuracy': results['vanilla_acc'],
        'attention_accuracy': results['attention_acc'],
        'improvement': results['attention_acc'] - results['vanilla_acc'],
        'relative_improvement': (results['attention_acc'] / results['vanilla_acc'] - 1) * 100
    })
    
    # Log figures
    wandb.log({
        'accuracy_by_length': wandb.Image(figures['length_fig']),
        'error_distribution': wandb.Image(figures['error_fig'])
    })
    
    # Log examples where attention helps
    attention_examples = results['attention_only_correct'].head(10)[
        ['Source', 'Target', 'Prediction_vanilla', 'Prediction_attention']
    ].values.tolist()
    
    wandb.log({
        'attention_helps_examples': wandb.Table(
            columns=['Source', 'Target', 'Vanilla Prediction', 'Attention Prediction'],
            data=attention_examples
        )
    })
    
    # Log examples where vanilla is better
    vanilla_examples = results['vanilla_only_correct'].head(10)[
        ['Source', 'Target', 'Prediction_vanilla', 'Prediction_attention']
    ].values.tolist()
    
    wandb.log({
        'vanilla_better_examples': wandb.Table(
            columns=['Source', 'Target', 'Vanilla Prediction', 'Attention Prediction'],
            data=vanilla_examples
        )
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare vanilla and attention models')
    parser.add_argument('--vanilla_model_path', type=str, required=True, help='Path to vanilla model checkpoint')
    parser.add_argument('--attention_model_path', type=str, required=True, help='Path to attention model checkpoint')
    parser.add_argument('--data_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/', 
                        help='Path to data directory')
    parser.add_argument('--wandb_project', type=str, default='transliteration', 
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default=None, 
                        help='W&B run name')
    args = parser.parse_args()
    
    # Load data and create vocabularies
    train_path = f"{args.data_path}/hi.translit.sampled.train.tsv"
    test_path = f"{args.data_path}/hi.translit.sampled.test.tsv"
    
    train_df = pd.read_csv(train_path, delimiter='\t', names=['hi', 'en', '_'])
    test_df = pd.read_csv(test_path, delimiter='\t', names=['hi', 'en', '_'])
    
    src_vocab = create_vocab(train_df['en'])
    tgt_vocab = create_vocab(train_df['hi'])
    
    # Create reverse mappings
    idx2src = {idx: char for char, idx in src_vocab.items()}
    idx2tgt = {idx: char for char, idx in tgt_vocab.items()}
    
    # Load models
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
    
    # Initialize models
    from models.seq2seq import Seq2Seq
    from models.attention_seq2seq import AttentionSeq2Seq
    
    vanilla_model = Seq2Seq(config).to(device)
    attention_model = AttentionSeq2Seq(config).to(device)
    
    # Load weights
    vanilla_model.load_state_dict(torch.load(args.vanilla_model_path, map_location=device))
    attention_model.load_state_dict(torch.load(args.attention_model_path, map_location=device))
    
    # Create test dataset and dataloader
    from src.dataset import TransliterationDataset, collate_fn
    test_dataset = TransliterationDataset(test_df, src_vocab, tgt_vocab)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    # Compare models
    comparison_results = compare_models(
        vanilla_model, attention_model, test_loader, src_vocab, tgt_vocab, idx2src, idx2tgt
    )
    
    # Plot results
    figures = plot_comparison_results(comparison_results)
    
    # Log to W&B
    log_comparison_to_wandb(comparison_results, figures, args.run_name)
    
    # Save predictions to CSV
    comparison_results['vanilla_preds'].to_csv('predictions_vanilla.csv', index=False)
    comparison_results['attention_preds'].to_csv('predictions_attention.csv', index=False)
    
    print("Comparison complete. Results saved to CSV and logged to W&B.")
