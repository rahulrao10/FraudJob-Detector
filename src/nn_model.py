import torch
import torch.nn as nn


class FraudNet(nn.Module):
    """Embedding -> (mean pool + max pool) -> concat hand-made features -> dense -> logit."""

    def __init__(self, vocab_size, emb_dim, n_feats):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.drop = nn.Dropout(0.4)
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 2 + n_feats, 64), nn.ReLU(),
            nn.Dropout(0.4), nn.Linear(64, 1),
        )

    def forward(self, ids, feats):
        mask = (ids != 0).unsqueeze(-1)
        e = self.emb(ids)
        mean = (e * mask).sum(1) / mask.sum(1).clamp(min=1)
        mx = e.masked_fill(~mask, -1e9).max(1).values
        h = torch.cat([self.drop(mean), self.drop(mx), feats], dim=1)
        return self.fc(h).squeeze(-1)
