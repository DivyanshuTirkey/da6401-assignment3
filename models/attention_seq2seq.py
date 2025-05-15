import torch
import torch.nn as nn
import random

class AttentionSeq2Seq(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(
            config['input_vocab_size'],
            config['embedding_dim'],
            config['hidden_dim'],
            config['num_encoding_layers'],
            config['dropout'],
            config['cell_type']
        )
        self.decoder = AttentionDecoder(
            config['output_vocab_size'],
            config['embedding_dim'],
            config['hidden_dim'],
            config['num_decoding_layers'],
            config['dropout'],
            config['cell_type']
        )
        self.device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.teacher_forcing_ratio = config.get('teacher_forcing_ratio', 0.5)
        self.cell_type = config['cell_type'].lower()
        self.config = config
        
    def forward(self, src, trg, teacher_forcing=1):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_vocab_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # Tensor to store attention weights
        attentions = torch.zeros(batch_size, trg_len, src.shape[1]).to(self.device)
        
        # Encode source sequence
        encoder_outputs, hidden, cell = self.encoder(src)
        
        # Adjust hidden state dimensions if needed
        enc_layers = self.config['num_encoding_layers']
        dec_layers = self.config['num_decoding_layers']
        hidden_size = self.config['hidden_dim']
        
        if enc_layers != dec_layers:
            if self.cell_type != 'lstm':
                # Case 1: Encoder has more layers - take only what we need
                if enc_layers > dec_layers:
                    hidden = hidden[:dec_layers]
                # Case 2: Decoder has more layers - pad with zeros
                else:
                    padding = torch.zeros(dec_layers - enc_layers, batch_size, hidden_size).to(self.device)
                    hidden = torch.cat([hidden, padding], dim=0)
            else:  # LSTM case
                if enc_layers > dec_layers:
                    hidden = hidden[:dec_layers]
                    cell = cell[:dec_layers]
                else:
                    padding = torch.zeros(dec_layers - enc_layers, batch_size, hidden_size).to(self.device)
                    hidden = torch.cat([hidden, padding], dim=0)
                    cell = torch.cat([cell, padding], dim=0)
        
        # First input to decoder is <SOS> token
        input = trg[:, 0]
        
        for t in range(1, trg_len):
            # Get decoder output
            output, hidden, cell, attn_weights = self.decoder(input, hidden, cell, encoder_outputs)
            
            # Store prediction and attention weights
            outputs[:, t, :] = output
            attentions[:, t, :] = attn_weights.squeeze(1)
            
            # Teacher forcing
            teacher_force = random.random() < self.teacher_forcing_ratio * teacher_forcing
            
            # Get highest predicted token
            top1 = output.argmax(1)
            
            # Next input is either ground truth or predicted token
            input = trg[:, t] if teacher_force else top1
            
        return outputs, attentions
