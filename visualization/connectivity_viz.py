import numpy as np
import torch
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show, output_notebook, save, output_file
from bokeh.models import ColumnDataSource, CustomJS, TapTool
from bokeh.layouts import column
import wandb

def compute_gradient_connectivity(model, src_text, src_vocab, tgt_vocab, idx2tgt, max_len=50):
    """
    Compute gradient-based connectivity between input and output characters.
    """
    model.train()
    
    # Convert to indices
    src_indices = text_to_indices(src_text, src_vocab)
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    
    # Get embeddings with gradient tracking
    src_emb = model.encoder.embedding(src_tensor)
    src_emb.retain_grad()  # Keep gradients for this tensor
    
    # Get encoder outputs
    if isinstance(model.encoder.rnn, torch.nn.LSTM):
        encoder_outputs, (hidden, cell) = model.encoder.rnn(src_emb)
    else:  # GRU or RNN
        encoder_outputs, hidden = model.encoder.rnn(src_emb)
        cell = None
    
    # Adjust hidden state dimensions if needed
    enc_layers = model.config['num_encoding_layers']
    dec_layers = model.config['num_decoding_layers']
    hidden_size = model.config['hidden_dim']
    batch_size = 1
    
    if enc_layers != dec_layers:
        if cell is None:  # GRU or RNN
            if enc_layers > dec_layers:
                hidden = hidden[:dec_layers]
            else:
                padding = torch.zeros(dec_layers - enc_layers, batch_size, hidden_size).to(device)
                hidden = torch.cat([hidden, padding], dim=0)
        else:  # LSTM
            if enc_layers > dec_layers:
                hidden = hidden[:dec_layers]
                cell = cell[:dec_layers]
            else:
                padding = torch.zeros(dec_layers - enc_layers, batch_size, hidden_size).to(device)
                hidden = torch.cat([hidden, padding], dim=0)
                cell = torch.cat([cell, padding], dim=0)
    
    # Start decoding with <SOS> token
    trg_idx = [tgt_vocab['<SOS>']]
    gradient_list = []
    
    for _ in range(max_len):
        # Clear previous gradients
        model.zero_grad()
        
        # Get current decoder input
        trg_tensor = torch.LongTensor([trg_idx[-1]]).to(device)
        
        # Forward pass through decoder
        if hasattr(model.decoder, 'attention'):
            # For attention model
            output, hidden, cell, _ = model.decoder(trg_tensor, hidden, cell, encoder_outputs)
        else:
            # For vanilla model
            if cell is not None:  # LSTM
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell)
            else:  # GRU or RNN
                output, hidden = model.decoder(trg_tensor, hidden)
        
        # Get predicted token
        pred_token = output.argmax(1).item()
        
        # Compute gradients with respect to the predicted token
        output[0, pred_token].backward(retain_graph=True)
        
        # Get gradients for embedding
        if src_emb.grad is not None:
            # Sum across embedding dimension and square (for magnitude)
            grad_magnitude = src_emb.grad.pow(2).sum(dim=2).squeeze(0).detach().cpu().numpy()
            gradient_list.append(grad_magnitude)
        else:
            # Fallback if no gradients
            gradient_list.append(np.ones(len(src_indices)) / len(src_indices))
        
        # Reset gradients for next iteration
        if src_emb.grad is not None:
            src_emb.grad.zero_()
        
        # Stop if <EOS> token
        if pred_token == tgt_vocab['<EOS>']:
            break
        
        trg_idx.append(pred_token)
    
    # Convert indices to characters
    trg_tokens = [idx2tgt[i] for i in trg_idx if i not in [tgt_vocab['<SOS>'], tgt_vocab['<EOS>'], tgt_vocab['<PAD>']]]
    translation = ''.join(trg_tokens)
    
    # Create connectivity matrix
    connectivity = np.zeros((len(trg_tokens), len(src_text)))
    for i, grad in enumerate(gradient_list[:len(trg_tokens)]):
        if i < len(trg_tokens):
            connectivity[i, :len(src_text)] = grad[:len(src_text)]
            # Normalize each row
            if np.sum(connectivity[i]) > 0:
                connectivity[i] = connectivity[i] / np.max(connectivity[i])
    
    return translation, connectivity

def create_bokeh_character_boxes_plot(model, src_text, src_vocab, tgt_vocab, idx2tgt):
    """
    Create an interactive plot with character boxes:
    - Output characters in boxes on top row
    - Input characters in boxes on bottom row
    - Clicking output box highlights input boxes based on connectivity
    """
    output_notebook()
    
    # Compute connectivity
    translation, connectivity = compute_gradient_connectivity(model, src_text, src_vocab, tgt_vocab, idx2tgt)
    
    # Prepare data for character boxes
    src_chars = list(src_text)
    tgt_chars = list(translation)
    
    # Create data sources for output and input characters
    output_data = {
        'x': list(range(len(tgt_chars))),
        'y': [0] * len(tgt_chars),
        'char': tgt_chars,
        'index': list(range(len(tgt_chars)))
    }
    output_source = ColumnDataSource(data=output_data)
    
    input_data = {
        'x': list(range(len(src_chars))),
        'y': [0] * len(src_chars),
        'char': src_chars,
        'color': ['#e6e6e6'] * len(src_chars),  # Light gray default
        'alpha': [1.0] * len(src_chars)
    }
    input_source = ColumnDataSource(data=input_data)
    
    # Output character boxes (top)
    output_plot = figure(
        title="Output Characters (Devanagari)",
        x_range=(-0.5, len(tgt_chars) - 0.5),
        y_range=(-0.5, 0.5),
        width=600, height=100,
        tools="tap",  # Only tap tool for interactivity
        toolbar_location=None
    )
    
    # Input character boxes (bottom)
    input_plot = figure(
        title="Input Characters (Latin)",
        x_range=(-0.5, len(src_chars) - 0.5),
        y_range=(-0.5, 0.5),
        width=600, height=100,
        tools="",
        toolbar_location=None
    )
    
    # Output rectangles and text
    output_rect = output_plot.rect(
        x='x', y='y', width=0.9, height=0.9,
        source=output_source,
        fill_color="#64b5f6",  # Light blue
        line_color="black",
        line_width=2
    )
    output_text = output_plot.text(
        x='x', y='y', text='char',
        source=output_source,
        text_align="center",
        text_baseline="middle",
        text_font_size="16px"
    )
    
    # Input rectangles and text
    input_rect = input_plot.rect(
        x='x', y='y', width=0.9, height=0.9,
        source=input_source,
        fill_color='color',
        line_color="black",
        line_width=2
    )
    input_text = input_plot.text(
        x='x', y='y', text='char',
        source=input_source,
        text_align="center",
        text_baseline="middle",
        text_font_size="16px"
    )
    
    # Interactivity: clicking output box highlights input boxes
    tap_callback = CustomJS(args=dict(
        input_source=input_source,
        connectivity=connectivity.tolist()
    ), code="""
        var index = cb_obj.selected.indices[0];
        if (index !== undefined) {
            var conn_row = connectivity[index];
            var colors = input_source.data['color'];
            for (let i = 0; i < colors.length; i++) {
                var weight = conn_row[i];
                var r = 255;
                var g = Math.max(0, Math.round(255 * (1 - weight)));
                var b = Math.max(0, Math.round(255 * (1 - weight)));
                colors[i] = `rgb(${r}, ${g}, ${b})`;
            }
            input_source.change.emit();
        }
    """)
    output_source.selected.js_on_change('indices', tap_callback)
    
    # Remove grid lines and axes
    output_plot.grid.grid_line_color = None
    output_plot.axis.visible = False
    input_plot.grid.grid_line_color = None
    input_plot.axis.visible = False
    
    layout = column(output_plot, input_plot, sizing_mode="stretch_width")
    return layout

def log_interactive_plot_to_wandb(model, src_text, src_vocab, tgt_vocab, idx2tgt, run=None):
    """
    Create an interactive connectivity plot and log it to Weights & Biases.
    """
    # Create the interactive plot
    interactive_plot = create_bokeh_character_boxes_plot(model, src_text, src_vocab, tgt_vocab, idx2tgt)
    
    # Save the plot to an HTML file
    output_file(f"connectivity_plot_{src_text}.html")
    save(interactive_plot)
    
    # Initialize W&B if not already initialized
    if run is None and wandb.run is None:
        wandb.init(project="transliteration")
    
    # Log the HTML file to W&B
    if run:
        run.log({f"connectivity_{src_text}": wandb.Html(f"connectivity_plot_{src_text}.html")})
    else:
        wandb.log({f"connectivity_{src_text}": wandb.Html(f"connectivity_plot_{src_text}.html")})
    
    print(f"Interactive plot for '{src_text}' saved to W&B")

def log_connectivity_grid_to_wandb(model, test_examples, src_vocab, tgt_vocab, idx2tgt, run=None):
    """
    Create and log a grid of connectivity visualizations to W&B.
    """
    # Create a table to hold all visualizations
    columns = ["example", "visualization"]
    data = []
    
    # Generate and save plots for each example
    for i, src_text in enumerate(test_examples):
        # Create and save the plot
        interactive_plot = create_bokeh_character_boxes_plot(model, src_text, src_vocab, tgt_vocab, idx2tgt)
        output_file(f"connectivity_plot_{i}.html")
        save(interactive_plot)
        
        # Add to data
        data.append([src_text, wandb.Html(f"connectivity_plot_{i}.html")])
    
    # Initialize W&B if not already initialized
    if run is None and wandb.run is None:
        wandb.init(project="transliteration")
    
    # Create and log the table
    table = wandb.Table(columns=columns, data=data)
    if run:
        run.log({"connectivity_grid": table})
    else:
        wandb.log({"connectivity_grid": table})
    
    print(f"Connectivity grid with {len(test_examples)} examples logged to W&B")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create connectivity visualizations')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model_type', type=str, choices=['vanilla', 'attention'], required=True, 
                        help='Type of model')
    parser.add_argument('--data_path', type=str, default='dakshina_dataset_v1.0/hi/lexicons/', 
                        help='Path to data directory')
    parser.add_argument('--wandb_project', type=str, default='transliteration', 
                        help='W&B project name')
    parser.add_argument('--example', type=str, default='ankur', 
                        help='Example to visualize')
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
    if args.model_type == 'attention':
        from models.attention_seq2seq import AttentionSeq2Seq
        model = AttentionSeq2Seq(config).to(device)
    else:
        from models.seq2seq import Seq2Seq
        model = Seq2Seq(config).to(device)
    
    # Load weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Create and log visualization
    wandb.init(project=args.wandb_project)
    log_interactive_plot_to_wandb(model, args.example, src_vocab, tgt_vocab, idx2tgt)
    
    # Create and log grid for multiple examples
    test_examples = ["ankur", "uncle", "ankganit", "hindi", "bharat", "computer", "india", "delhi", "mumbai"]
    log_connectivity_grid_to_wandb(model, test_examples, src_vocab, tgt_vocab, idx2tgt)
