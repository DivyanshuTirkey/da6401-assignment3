import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embedding_dim, hidden_dim, num_layers, dropout, cell_type):
        super().__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.cell_type = cell_type.lower()
        
        if self.cell_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                              dropout=dropout if num_layers > 1 else 0, batch_first=True)
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, 
                             dropout=dropout if num_layers > 1 else 0, batch_first=True)
        else:  # rnn
            self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, 
                             dropout=dropout if num_layers > 1 else 0, batch_first=True)
        
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
        
    def forward(self, src):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]
        
        if self.cell_type == "lstm":
            outputs, (hidden, cell) = self.rnn(embedded)
            return outputs, hidden, cell
        else:
            outputs, hidden = self.rnn(embedded)
            return outputs, hidden, None
