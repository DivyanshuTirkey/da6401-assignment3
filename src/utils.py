import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os

def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def create_vocab(texts, special_tokens=True):
    """Create vocabulary from texts"""
    chars = set()
    for text in texts:
        for char in str(text):
            chars.add(char)
    
    # Create vocabulary dictionary
    if special_tokens:
        vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    else:
        vocab = {}
    
    for i, char in enumerate(sorted(list(chars))):
        vocab[char] = i + 4 if special_tokens else i
    
    return vocab

def text_to_indices(text, vocab):
    """Convert text to indices"""
    indices = [vocab['<SOS>']]
    for char in str(text):
        if char in vocab:
            indices.append(vocab[char])
        elif char.lower() in vocab:
            indices.append(vocab[char.lower()])
        else:
            indices.append(vocab['<UNK>'])
    indices.append(vocab['<EOS>'])
    return indices

def get_devanagari_font():
    """Get a font that supports Devanagari script"""
    # Try to find a suitable font
    font_paths = [
        '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf',
        '/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf',
        '/usr/share/fonts/truetype/fonts-deva-extra/samanata.ttf'
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            return FontProperties(fname=path)
    
    # If no specific font is found, return None and let matplotlib use default
    return None

def plot_attention(sentence, translation, attention, idx2src):
    """Plot attention weights"""
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Remove special tokens from sentence
    sentence_chars = [c for c in sentence]
    
    attention = attention[:len(translation), :len(sentence_chars)]
    
    ax.matshow(attention, cmap='Blues')
    
    ax.set_xticklabels([''] + sentence_chars, rotation=90)
    ax.set_yticklabels([''] + list(translation))
    
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))
    
    plt.tight_layout()
    return fig

def create_attention_grid(model, test_examples, src_vocab, tgt_vocab, idx2src, idx2tgt, rows=3, cols=3):
    """Create a grid of attention visualizations for test examples."""
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    examples = test_examples[:rows*cols]
    
    # Get Devanagari font
    font_prop = get_devanagari_font()
    
    for i, example in enumerate(examples):
        row = i // cols
        col = i % cols
        
        # Get translation and attention matrix
        translation, attention_matrix = translate_sentence(
            model, example, src_vocab, tgt_vocab, idx2tgt
        )
        
        # Get source and target tokens
        src_tokens = [c for c in example]
        tgt_tokens = [c for c in translation]
        
        # Plot attention matrix
        attention = np.array(attention_matrix)[:len(tgt_tokens), :len(src_tokens)]
        im = axes[row, col].imshow(attention, cmap='Blues')
        
        # Set labels
        axes[row, col].set_xticks(range(len(src_tokens)))
        axes[row, col].set_yticks(range(len(tgt_tokens)))
        axes[row, col].set_xticklabels(src_tokens, fontsize=8, rotation=90)
        
        # Use Devanagari font if available
        if font_prop:
            axes[row, col].set_yticklabels(tgt_tokens, fontsize=8, fontproperties=font_prop)
        else:
            axes[row, col].set_yticklabels(tgt_tokens, fontsize=8)
            
        axes[row, col].set_title(f"{example} → {translation}", fontsize=10)
    
    plt.tight_layout()
    return fig


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_computational_complexity(config):
    """
    Compute the number of operations in the model
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Dictionary with computational complexity for different cell types
    """
    m = config['embedding_dim']
    k = config['hidden_dim']
    T = 20  # Approximate sequence length
    V = max(config['input_vocab_size'], config['output_vocab_size'])
    
    # RNN complexity
    rnn_ops = 2 * T * (k**2 + k * m)
    
    # LSTM complexity (4 gates)
    lstm_ops = 8 * T * (k**2 + k * m)
    
    # GRU complexity (3 gates)
    gru_ops = 6 * T * (k**2 + k * m)
    
    return {
        'rnn': rnn_ops,
        'lstm': lstm_ops,
        'gru': gru_ops
    }

def compute_parameter_count(config):
    """
    Compute the number of parameters in the model
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Dictionary with parameter counts for different cell types
    """
    m = config['embedding_dim']
    k = config['hidden_dim']
    V = max(config['input_vocab_size'], config['output_vocab_size'])
    
    # Common parameters
    embedding_params = 2 * V * m  # Encoder and decoder embeddings
    output_params = k * V + V  # Output layer
    
    # RNN parameters
    rnn_params = 2 * (m * k + k**2 + k)  # Encoder and decoder
    
    # LSTM parameters (4 gates)
    lstm_params = 2 * (4 * (m * k + k**2) + 4 * k)  # Encoder and decoder
    
    # GRU parameters (3 gates)
    gru_params = 2 * (3 * (m * k + k**2) + 3 * k)  # Encoder and decoder
    
    return {
        'rnn': embedding_params + rnn_params + output_params,
        'lstm': embedding_params + lstm_params + output_params,
        'gru': embedding_params + gru_params + output_params
    }