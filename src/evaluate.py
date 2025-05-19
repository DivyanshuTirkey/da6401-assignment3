import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    
    # For exact match accuracy
    exact_match_correct = 0
    exact_match_total = 0
    
    # For character-level accuracy
    char_correct = 0
    char_total = 0
    
    with torch.no_grad():
        for src, trg in tqdm(dataloader, desc="Evaluating", leave=False):
            src, trg = src.to(model.device), trg.to(model.device)
            
            output, _ = model(src, trg, 0)  # Turn off teacher forcing
            
            # For loss calculation
            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)
            
            loss = criterion(output_flat, trg_flat)
            epoch_loss += loss.item()
            
            # Get predictions
            predictions = output.argmax(dim=2)
            
            # Calculate exact match and character-level accuracy
            for i in range(len(predictions)):
                pred_seq = predictions[i, 1:].cpu().numpy()  # Skip <SOS>
                target_seq = trg[i, 1:].cpu().numpy()  # Skip <SOS>
                
                # Get valid sequence (remove padding)
                valid_length = (target_seq != 0).sum()  # Assuming 0 is <PAD>
                pred_clean = pred_seq[:valid_length]
                target_clean = target_seq[:valid_length]
                
                # Check exact match
                if np.array_equal(pred_clean, target_clean):
                    exact_match_correct += 1
                exact_match_total += 1
                
                # Calculate character-level accuracy
                for j in range(valid_length):
                    if pred_seq[j] == target_seq[j]:
                        char_correct += 1
                    char_total += 1
    
    # Calculate metrics
    exact_match_accuracy = exact_match_correct / exact_match_total if exact_match_total > 0 else 0
    char_accuracy = char_correct / char_total if char_total > 0 else 0
    
    return {
        'loss': epoch_loss / len(dataloader),
        'exact_match_accuracy': exact_match_accuracy,
        'char_accuracy': char_accuracy
    }

def generate_predictions(model, dataloader, src_vocab, tgt_vocab, idx2tgt):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for src, trg in tqdm(dataloader, desc="Generating predictions"):
            src, trg = src.to(model.device), trg.to(model.device)
            
            # Forward pass
            output, _ = model(src, trg, 0)  # No teacher forcing
            
            # Get predictions
            pred_indices = output.argmax(dim=2)
            
            # Convert to characters
            for i in range(len(pred_indices)):
                src_seq = src[i].cpu().numpy()
                trg_seq = trg[i].cpu().numpy()
                pred_seq = pred_indices[i].cpu().numpy()
                
                # Remove padding and special tokens
                src_valid = [idx2src[idx] for idx in src_seq if idx not in [0, 1, 2, 3]]  # <PAD>, <SOS>, <EOS>, <UNK>
                trg_valid = [idx2tgt[idx] for idx in trg_seq if idx not in [0, 1, 2, 3]]
                pred_valid = [idx2tgt[idx] for idx in pred_seq if idx not in [0, 1, 2, 3]]
                
                # Join characters
                src_text = ''.join(src_valid)
                trg_text = ''.join(trg_valid)
                pred_text = ''.join(pred_valid)
                
                # Check if correct
                correct = (trg_text == pred_text)
                
                predictions.append({
                    'Source': src_text,
                    'Target': trg_text,
                    'Prediction': pred_text,
                    'Correct': correct
                })
    
    return pd.DataFrame(predictions)

def translate_sentence(model, sentence, src_vocab, tgt_vocab, idx2tgt, max_len=50):
    model.eval()
    
    # Convert to indices and add <SOS> and <EOS>
    indices = text_to_indices(sentence, src_vocab)
    src_tensor = torch.LongTensor(indices).unsqueeze(0).to(model.device)
    
    # Get encoder outputs
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        # Adjust hidden state dimensions if needed
        enc_layers = model.config['num_encoding_layers']
        dec_layers = model.config['num_decoding_layers']
        hidden_size = model.config['hidden_dim']
        
        if enc_layers != dec_layers:
            batch_size = 1  # Since we're translating one sentence
            if model.cell_type != 'lstm':
                if enc_layers > dec_layers:
                    hidden = hidden[:dec_layers]
                else:
                    padding = torch.zeros(dec_layers - enc_layers, batch_size, hidden_size).to(model.device)
                    hidden = torch.cat([hidden, padding], dim=0)
            else:  # LSTM case
                if enc_layers > dec_layers:
                    hidden = hidden[:dec_layers]
                    cell = cell[:dec_layers]
                else:
                    padding = torch.zeros(dec_layers - enc_layers, batch_size, hidden_size).to(model.device)
                    hidden = torch.cat([hidden, padding], dim=0)
                    cell = torch.cat([cell, padding], dim=0)
        
        # Start with <SOS> token
        trg_idx = [tgt_vocab['<SOS>']]
        attentions = []
        
        for _ in range(max_len):
            trg_tensor = torch.LongTensor([trg_idx[-1]]).to(model.device)
            
            # Forward pass through decoder
            if hasattr(model, 'decoder') and hasattr(model.decoder, 'attention'):
                # For attention model
                output, hidden, cell, attn_weights = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
                # Store attention weights
                attentions.append(attn_weights.squeeze().cpu().numpy())
            else:
                # For vanilla model
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            
            pred_token = output.argmax(1).item()
            
            # Stop if <EOS> token
            if pred_token == tgt_vocab['<EOS>']:
                break
            
            trg_idx.append(pred_token)
        
        # Convert indices to characters
        trg_tokens = [idx2tgt[i] for i in trg_idx if i not in [tgt_vocab['<SOS>'], tgt_vocab['<EOS>'], tgt_vocab['<PAD>']]]
        
    return ''.join(trg_tokens), attentions
