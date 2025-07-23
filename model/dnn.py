import torch
from torch import nn

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.champion_indices = torch.tensor([0, 12, 24, 36, 48, 101, 113, 125, 137, 149])
        self.champ_embedding = nn.Embedding(num_embeddings=171, embedding_dim=8)

        self.fc = nn.Sequential(
            nn.Linear(230, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        B = x.size(0)

        champion_ids = x[:, self.champion_indices]
        champion_ids = champion_ids.long()

        embedded = self.champ_embedding(champion_ids)
        embedded_flat = embedded.view(B, -1)

        mask = torch.ones(x.size(1), dtype=torch.bool, device=x.device)
        mask[self.champion_indices] = False
        x_non_cat = x[:, mask]

        x_final = torch.cat([x_non_cat, embedded_flat], dim=1)

        return self.fc(x_final)
