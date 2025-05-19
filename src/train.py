import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import pandas as pd
import random
import numpy as np

from models.seq2seq import Seq2Seq
from models.attention_seq2seq import AttentionSeq2Seq
from src.dataset import TransliterationDataset, collate_fn
from src.utils import create_vocab, text_to_indices
from src.evaluate import evaluate

def train_epoch(model, dataloader, optimizer, criterion, scaler=None, clip=1.0):
    model.train()
    epoch_loss = 0
    
    for src, trg in tqdm(dataloader, desc="Training", leave=False):
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler:
            with autocast():
                output, _ = model(src, trg)
                
                # Exclude <SOS> token
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)
                
                loss = criterion(output, trg)
                
            # Scale loss and backprop
            scaler.scale(loss).backward()
            
            # Unscale gradients and clip
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            # Update weights
            scaler.step(optimizer)
            scaler.update()
        else:
            output, _ = model(src, trg)
            
            # Exclude <SOS> token
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser(description='Train a transliteration model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_type', type=str, choices=['vanilla', 'attention'], default='vanilla', 
                        help='Type of model to train')
    parser.add_argument('--data_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/', 
                        help='Path to data directory')
    parser.add_argument('--wandb_project', type=str, default='transliteration', 
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, 
                        help='W&B entity name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_epochs', type=int, default=30, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    with open(args.config, 'r') as f:
        sweep_config = yaml.safe_load(f)
    
    # Data paths
    train_path = f"{args.data_path}/hi.translit.sampled.train.tsv"
    val_path = f"{args.data_path}/hi.translit.sampled.dev.tsv"
    test_path = f"{args.data_path}/hi.translit.sampled.test.tsv"
    
    # Load data
    train_df = pd.read_csv(train_path, delimiter='\t', names=['hi', 'en', '_'])
    val_df = pd.read_csv(val_path, delimiter='\t', names=['hi', 'en', '_'])
    test_df = pd.read_csv(test_path, delimiter='\t', names=['hi', 'en', '_'])
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create vocabularies
    src_vocab = create_vocab(train_df['en'])
    tgt_vocab = create_vocab(train_df['hi'])
    
    # Create reverse mappings for visualization
    idx2src = {idx: char for char, idx in src_vocab.items()}
    idx2tgt = {idx: char for char, idx in tgt_vocab.items()}
    
    print(f"Source vocabulary size: {len(src_vocab)}")
    print(f"Target vocabulary size: {len(tgt_vocab)}")
    
    # Initialize W&B sweep
    def train_with_config():
        # Initialize wandb run
        run = wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        
        # Get hyperparameters from sweep
        config = {
            'input_vocab_size': len(src_vocab),
            'output_vocab_size': len(tgt_vocab),
            'embedding_dim': wandb.config.embedding_dim,
            'hidden_dim': wandb.config.hidden_dim,
            'num_encoding_layers': wandb.config.num_encoding_layers,
            'num_decoding_layers': wandb.config.num_decoding_layers,
            'dropout': wandb.config.dropout,
            'cell_type': wandb.config.cell_type,
            'teacher_forcing_ratio': wandb.config.teacher_forcing_ratio,
            'learning_rate': wandb.config.learning_rate,
            'batch_size': wandb.config.batch_size,
            'device': device
        }
        
        # Create datasets
        train_dataset = TransliterationDataset(train_df, src_vocab, tgt_vocab)
        val_dataset = TransliterationDataset(val_df, src_vocab, tgt_vocab)
        test_dataset = TransliterationDataset(test_df, src_vocab, tgt_vocab)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Initialize model
        if args.model_type == 'attention':
            model = AttentionSeq2Seq(config).to(device)
        else:
            model = Seq2Seq(config).to(device)
        
        # Log model architecture
        wandb.watch(model, log='all')
        
        # Optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<PAD>'])
        
        # Mixed precision training
        scaler = GradScaler() if device.type == 'cuda' else None
        
        # Training loop
        best_exact_match = 0
        best_char_accuracy = 0
        patience_counter = 0
        
        for epoch in range(args.max_epochs):
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
            
            # Evaluate
            eval_metrics = evaluate(model, val_loader, criterion)
            
            # Log metrics
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': eval_metrics['loss'],
                'val_exact_match': eval_metrics['exact_match_accuracy'],
                'val_char_accuracy': eval_metrics['char_accuracy']
            })
            
            print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {eval_metrics['loss']:.4f}")
            print(f"Exact Match Accuracy: {eval_metrics['exact_match_accuracy']:.4f}, Char Accuracy: {eval_metrics['char_accuracy']:.4f}")
            
            # Save best model
            if eval_metrics['exact_match_accuracy'] > best_exact_match:
                best_exact_match = eval_metrics['exact_match_accuracy']
                best_char_accuracy = eval_metrics['char_accuracy']
                model_path = f'best_{args.model_type}_model_{wandb.run.id}.pt'
                torch.save(model.state_dict(), model_path)
                wandb.save(model_path)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model for testing
        model.load_state_dict(torch.load(model_path))
        
        # Test evaluation
        test_metrics = evaluate(model, test_loader, criterion)
        
        # Log test metrics
        wandb.log({
            'test_loss': test_metrics['loss'],
            'test_exact_match': test_metrics['exact_match_accuracy'],
            'test_char_accuracy': test_metrics['char_accuracy']
        })
        
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Exact Match Accuracy: {test_metrics['exact_match_accuracy']:.4f}")
        print(f"Test Char Accuracy: {test_metrics['char_accuracy']:.4f}")
        
        return model
    
    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=args.wandb_project, entity=args.wandb_entity)
    
    # Start the sweep agent
    wandb.agent(sweep_id, train_with_config, count=40)  # Run 40 experiments

if __name__ == "__main__":
    main()
