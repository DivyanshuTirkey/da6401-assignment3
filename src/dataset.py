import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TransliterationDataset(Dataset):
    def __init__(self, dataframe, src_vocab, tgt_vocab):
        self.dataframe = dataframe
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        src_text = self.dataframe.iloc[idx]['en']
        tgt_text = self.dataframe.iloc[idx]['hi']
        
        src_indices = text_to_indices(src_text, self.src_vocab)
        tgt_indices = text_to_indices(tgt_text, self.tgt_vocab)
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

def text_to_indices(text, vocab):
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

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src, tgt in batch:
        # Safety check for index bounds
        src = torch.clamp(src, 0, len(src_vocab)-1)
        tgt = torch.clamp(tgt, 0, len(tgt_vocab)-1)
        
        # Pad or truncate to max lengths
        src = src[:20]  # Max source length is 20
        tgt = tgt[:19]  # Max target length is 19
        
        src_batch.append(src)
        tgt_batch.append(tgt)
    
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab['<PAD>'])
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab['<PAD>'])
    
    return src_batch, tgt_batch
