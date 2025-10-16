
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils.data import Dataset

# Dataset + Masked Modeling
class BERT4RecDataset(Dataset):
    def __init__(self, user_seqs, n_items, max_len=50, mask_prob=0.15):
        self.user_seqs = user_seqs
        self.users = list(user_seqs.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.masked_id = n_items + 1

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        seq = self.user_seqs[u][-self.max_len:]
        seq = [0]*(self.max_len - len(seq)) + seq

        seq = torch.tensor(seq, dtype=torch.long)

        # Masking
        masked_seq = seq.clone()
        labels = torch.full_like(seq, -100)  # ignore index
        prob = torch.rand(seq.size())
        mask = (prob < self.mask_prob) & (seq != 0)
        masked_seq[mask] = self.masked_id
        labels[mask] = seq[mask]

        return masked_seq, labels

# Model
class BERT4Rec(nn.Module):
    def __init__(self, n_items, hidden_dim=64, max_len=50,
                 num_layers=2, num_heads=2, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items+2, hidden_dim, padding_idx=0)  # +1 for MASK
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        self.masked_id = n_items + 1

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.n_items = n_items

    def forward(self, seq):
        B, L = seq.shape
        pos_ids = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(pos_ids)

        x = self.encoder(x)
        x = self.norm(x)
        return x

    def predict(self, seq, candidates, device=None):
        # adjust seq length
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        elif len(seq) < self.max_len:
            seq = [0] * (self.max_len - len(seq)) + seq
        seq[-1] = self.masked_id  # mask last position
        
        # convert to tensor
        seq        = torch.tensor([seq],      dtype=torch.long)
        candidates = torch.tensor(candidates, dtype=torch.long)
        if device:
            seq = seq.to(device)
            candidates = candidates.to(device)

        # predict
        x = self.forward(seq)  # [B, L, H]
        last_hidden = x[:, -1, :]  # [B, H]

        cand_emb = self.item_emb(candidates).unsqueeze(0)  # [1, C, H] (or [B, C, H])
        last_hidden = last_hidden.unsqueeze(-1)  # [B, H, 1]

        scores = torch.bmm(cand_emb, last_hidden).squeeze(-1)  # [B, C]
        return scores
