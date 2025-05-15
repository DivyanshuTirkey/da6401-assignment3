import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionDecoder(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_dim, num_layers, dropout, cell_type):
        super().__init__()
        self.output_vocab_size = output_vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.cell_type = cell_type.lower()
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attention_combine = nn.Linear(hidden_dim + embedding_dim, embedding_dim)
        
        # RNN layer
        if cell_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                              dropout=dropout if num_layers > 1 else 0, batch_first=True)
        elif cell_type == "gru":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, 
                             dropout=dropout if num_layers > 1 else 0, batch_first=True)
        else:  # rnn
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, 
                             dropout=dropout if num_layers > 1 else 0, batch_first=True)
        
        self.fc_out = nn.Linear(hidden_dim, output_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Apply weight initialization
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
    def forward(self, input, hidden, cell, encoder_outputs):
        # input: [batch_size]
        # hidden: [num_layers, batch_size, hidden_dim]
        # encoder_outputs: [batch_size, src_len, hidden_dim]
        
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]
        
        # Calculate attention weights
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)
        
        # Use the last layer of hidden state for attention
        attn_hidden = hidden[-1].unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # Repeat for each encoder output
        attn_hidden = attn_hidden.repeat(1, src_len, 1)  # [batch_size, src_len, hidden_dim]
        
        # Concatenate encoder outputs and hidden state
        energy = torch.cat((encoder_outputs, attn_hidden), dim=2)  # [batch_size, src_len, 2*hidden_dim]
        energy = self.attention(energy)  # [batch_size, src_len, hidden_dim]
        energy = torch.tanh(energy)
        
        # Calculate attention weights
        attn_weights = torch.sum(energy, dim=2)  # [batch_size, src_len]
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(1)  # [batch_size, 1, src_len]
        
        # Apply attention weights to encoder outputs
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch_size, 1, hidden_dim]
        
        # Combine embedded input and context vector
        rnn_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, emb_dim + hidden_dim]
        rnn_input = self.attention_combine(rnn_input)  # [batch_size, 1, emb_dim]
        
        # Pass through RNN
        if self.cell_type == "lstm":
            output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        else:
            output, hidden = self.rnn(rnn_input, hidden)
            cell = None
            
        # Generate output
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_vocab_size]
        
        return prediction, hidden, cell, attn_weights
