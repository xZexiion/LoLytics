import torch
from torch import nn

class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.champion_indices = torch.tensor([0, 12, 24, 36, 48, 101, 113, 125, 137, 149])
        self.champ_embedding = nn.Embedding(num_embeddings=171, embedding_dim=8)

        self.fc = nn.Sequential(
            nn.Linear(273, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.kill_indices = torch.tensor([0, 11, 22, 33, 44, 96, 107, 118, 129, 140])
        self.death_indices = torch.tensor([1, 12, 23, 34, 45, 97, 108, 119, 130, 141])
        self.assist_indices = torch.tensor([2, 13, 24, 35, 46, 98, 109, 120, 131, 142])
        self.gold_indices = torch.tensor([6, 17, 28, 39, 50, 102, 113, 124, 135, 146])
        self.cs_indices = torch.tensor([8, 19, 30, 41, 52, 104, 115, 126, 137, 148])
        self.baron_indices = torch.tensor([3, 14, 25, 36, 47, 99, 110, 121, 132, 143])
        self.elder_indices = torch.tensor([4, 15, 26, 37, 48, 100, 111, 122, 133, 144])
        self.death_timer_indices = torch.tensor([5, 16, 27, 38, 49, 101, 112, 123, 134, 145])
        self.level_indices = torch.tensor([7, 18, 29, 40, 51, 103, 114, 125, 136, 147])
        self.x_indices = torch.tensor([9, 20, 31, 42, 53, 105, 116, 127, 138, 149])
        self.y_indices = torch.tensor([10, 21, 32, 43, 54, 106, 117, 128, 139, 150])

    def normalize(self, x):
        x[:, self.kill_indices] = (x[:, self.kill_indices] - 2.686) / 3.281
        x[:, self.death_indices] = (x[:, self.death_indices] - 2.698) / 2.754
        x[:, self.assist_indices] = (x[:, self.assist_indices] - 3.259) / 4.216
        x[:, self.gold_indices] = (x[:, self.gold_indices] - 5840.304) / 4262.559
        x[:, self.cs_indices] = (x[:, self.cs_indices] - 88.101) / 74.53
        x[:, self.baron_indices] /= 3*60
        x[:, self.elder_indices] /= 2*60+30
        x[:, self.death_timer_indices] /= 79
        x[:, self.level_indices] /= 18
        x[:, self.x_indices] /= 14500
        x[:, self.y_indices] /= 14500

    def forward(self, x):
        B = x.size(0)

        champion_ids = x[:, self.champion_indices]
        champion_ids = champion_ids.long()

        embedded = self.champ_embedding(champion_ids)
        embedded_flat = embedded.view(B, -1)

        mask = torch.ones(x.size(1), dtype=torch.bool, device=x.device)
        mask[self.champion_indices] = False
        x_non_cat = x[:, mask].float()

        self.normalize(x_non_cat)
        x_non_cat[:, -1] /= 50

        x_final = torch.cat([x_non_cat, embedded_flat], dim=1)

        return self.fc(x_final)
