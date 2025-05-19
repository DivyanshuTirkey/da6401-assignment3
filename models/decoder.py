import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_vocab_size, embedding_dim, hidden_dim, num_layers, dropout, cell_type):
        super().__init__()
        self.output_vocab_size = output_vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.cell_type = cell_type.lower()
        
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
        
    def forward(self, input, hidden, cell):
        # input: [batch_size]
        # hidden: [num_layers, batch_size, hidden_dim]
        
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]
        
        # Pass through RNN
        if self.cell_type == "lstm":
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        else:
            output, hidden = self.rnn(embedded, hidden)
            cell = None
            
        # Generate output
        prediction = self.fc_out(output.squeeze(1))  # [batch_size, output_vocab_size]
        
        return prediction, hidden, cell
