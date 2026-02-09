import torch
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(self, vocab: int = 64, seq_len: int = 32, d_model: int = 192, n_heads: int = 4, n_layers: int = 2, num_classes: int = 64):
        super().__init__()
        self.vocab = vocab
        self.seq_len = seq_len
        self.num_classes = num_classes

        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, T]
        h = self.tok(x) + self.pos[:, : x.size(1), :]
        h = self.encoder(h)
        # use CLS-like pooling: mean over time
        h = h.mean(dim=1)
        return self.head(h)
