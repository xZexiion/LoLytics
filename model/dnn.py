import torch
from torch import nn

def normalize():

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.champion_indices = torch.tensor([0, 12, 24, 36, 48, 101, 113, 125, 137, 149])
        self.champ_embedding = nn.Embedding(num_embeddings=171, embedding_dim=8)

        self.fc = nn.Sequential(
            nn.Linear(272, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def normalize(x):
        for team_offset in [0, 11*5+24+3+11+3]:
            for j in range(5):
                player_offset = j * 11
                offset = team_offset + player_offset

                x[:, offset] = (x[:, offset] - 2.686) / 3.281
                x[:, offset+1] = (x[:, offset+1] - 2.698) / 2.754
                x[:, offset+2] = (x[:, offset+2] - 3.259) / 4.216
                x[:, offset+6] = (x[:, offset+6] - 5840.304) / 4262.559
                x[:, offset+8] = (x[:, offset+8] - 88.101) / 74.53

                x[:, offset+3] /= 3*60
                x[:, offset+4] /= 2*60+30
                x[:, offset+5] /= 79
                x[:, offset+7] /= 18
                x[:, offset+9] /= 14500
                x[:, offset+10] /= 14500
        
    def forward(self, x):
        B = x.size(0)

        champion_ids = x[:, self.champion_indices]
        champion_ids = champion_ids.long()

        embedded = self.champ_embedding(champion_ids)
        embedded_flat = embedded.view(B, -1)

        mask = torch.ones(x.size(1), dtype=torch.bool, device=x.device)
        mask[self.champion_indices] = False
        x_non_cat = x[:, mask]

        self.normalize(x_non_cat)

        x_final = torch.cat([x_non_cat, embedded_flat], dim=1)

        return self.fc(x_final)
