
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import random

# Dataset + Negative Sampling
class SASRecDataset(Dataset):
    def __init__(self, user_train, n_items, num_negatives=1, max_len=50):
        self.user_train = user_train
        self.users = list(user_train.keys())
        self.n_items = n_items
        self.num_negatives = num_negatives
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_train[user]

        if len(seq) < 2:
            return torch.zeros(self.max_len, dtype=torch.long), torch.tensor(0, dtype=torch.long), torch.tensor(0, dtype=torch.long)

        # Randomly choose a cut point
        cut = random.randint(1, len(seq) - 1)
        prefix, target = seq[:cut], seq[cut]  # target is the next item

        # Pad prefix
        seq_padded = [0] * (self.max_len - len(prefix)) + prefix[-self.max_len:]
        seq_tensor = torch.tensor(seq_padded, dtype=torch.long)  # training sequence

        # Negative samples
        negatives = []
        for _ in range(self.num_negatives):
            neg = random.randint(1, self.n_items)
            while neg in seq:
                neg = random.randint(1, self.n_items)
            negatives.append(neg)
        negatives = negatives[0] #if self.num_negatives == 1 else negatives

        return seq_tensor, torch.tensor(target, dtype=torch.long), torch.tensor(negatives, dtype=torch.long)

# Model
class SASRec(nn.Module):
    def __init__(self, n_items, hidden_dim=64, max_len=50, n_heads=2, n_layers=2, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items+1, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.hidden_dim = hidden_dim
        self.max_len = max_len

    def forward(self, seq):
        device = seq.device
        positions = torch.arange(self.max_len, device=device).unsqueeze(0)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(x)

        mask = (seq == 0)
        x = self.encoder(x, src_key_padding_mask=mask)

        out = x[:, -1, :]  # last position representation
        return out

    def predict(self, seq, candidates, device=None):
        # adjust seq length
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        elif len(seq) < self.max_len:
            seq = [0] * (self.max_len - len(seq)) + seq
        
        # convert to tensor
        seq        = torch.tensor([seq],      dtype=torch.long)
        candidates = torch.tensor(candidates, dtype=torch.long)
        if device:
            seq = seq.to(device)
            candidates = candidates.to(device)
        seq = seq.repeat(len(candidates), 1)
        
        # predict
        seq_repr  = self.forward(seq)          # [B, H]
        item_repr = self.item_emb(candidates)  # [B, H]
        return (seq_repr * item_repr).sum(dim=-1)
