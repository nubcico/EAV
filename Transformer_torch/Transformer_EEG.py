import os
import pickle
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, qkv_dim: int):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_dim = qkv_dim

        # One linear projection per temporal filter
        self.value_proj = nn.ModuleList(
            [nn.Linear(30, 1, bias=False) for _ in range(40)]
        )

    def forward(self, x):
        values = []
        for i in range(40):
            x_i = x[:, i].permute(0, 2, 1)  # (B, T, 30)
            v = self.value_proj[i](x_i)    # (B, T, 1)
            values.append(v)

        return torch.cat(values, dim=-1)  # (B, T, 40)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, qkv_dim: int):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(self.head_dim, qkv_dim, bias=False)
        self.W_k = nn.Linear(self.head_dim, qkv_dim, bias=False)
        self.W_v = nn.Linear(self.head_dim, qkv_dim, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        outputs = []
        residuals = []

        for h in range(self.num_heads):
            x_h = x[:, h]  # (B, T, head_dim)

            Q = self.W_q(x_h)
            K = self.W_k(x_h)
            V = self.W_v(x_h)

            attn = torch.matmul(Q, K.transpose(-1, -2)) / (self.head_dim ** 0.5)
            attn = F.softmax(attn, dim=-1)

            outputs.append(torch.matmul(attn, V))
            residuals.append(V)

        out = torch.cat(outputs, dim=-1)
        res = torch.cat(residuals, dim=-1)

        return out + res


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, expansion: int = 4, drop_p: float = 0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * expansion),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(embed_dim * expansion, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, qkv_dim: int, drop_p: float = 0.5):
        super().__init__()

        self.attn = MultiHeadAttention(embed_dim, num_heads, qkv_dim)
        self.ffn = FeedForwardBlock(embed_dim, drop_p=drop_p)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(drop_p)

    def forward(self, x):
        x = x + self.dropout(self.norm1(self.attn(x)))
        x = x + self.dropout(self.norm2(self.ffn(x)))
        return x


class ShallowConvNet(nn.Module):
    def __init__(
        self,
        nb_classes: int,
        chans: int = 30,
        samples: int = 500,
        dropout: float = 0.5,
        num_layers: int = 12,
    ):
        super().__init__()

        self.conv = nn.Conv2d(1, 40, (1, 13), bias=False)
        self.pool = nn.AvgPool2d((1, 35), stride=(1, 7))
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm2d(40)

        self.embedding = PatchEmbedding(embed_dim=40, num_heads=1, qkv_dim=40)
        self.transformer = nn.ModuleList(
            [TransformerLayer(40, 1, 40, dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(2600, nb_classes, bias=False)

    def forward(self, x):
        x = self.conv(x)

        v = self.embedding(x)
        for layer in self.transformer:
            v = layer(v)

        x = v.permute(0, 2, 1).unsqueeze(2)
        x = self.bn(x)

        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, 1e-7, 1e4))

        x = x.squeeze(2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)

        return F.softmax(self.fc(x), dim=1)


class TrainerUni:
    def __init__(
        self,
        model,
        data,
        lr=1e-3,
        batch_size=32,
        epochs=10,
        subject=0,
        device=None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tr_x, tr_y, te_x, te_y = data
        self.train_loader = self._loader(tr_x, tr_y, batch_size, True)
        self.test_loader = self._loader(te_x, te_y, batch_size, False)

        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.epochs = epochs
        self.subject = subject

    def _loader(x, y, batch_size, shuffle):
        return DataLoader(
            TensorDataset(x, y),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                out = self.model(x)
                loss = self.criterion(out, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    self.model.fc.weight.data = torch.renorm(
                        self.model.fc.weight.data, p=2, dim=0, maxnorm=0.5
                    )

            acc = self.validate()
            if epoch == self.epochs - 1:
                with open("eeg_results_new_shallow_.txt", "a") as f:
                    f.write(f"Subject {self.subject} | Accuracy: {acc:.4f}\n")

    def validate(self):
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")
        return acc

if __name__ == "__main__":
    data_path = r"D:\input images\EEG"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for subject in range(1, 2):
        file = os.path.join(data_path, f"subject_{subject:02d}_eeg.pkl")
        if not os.path.isfile(file):
            continue

        with open(file, "rb") as f:
            tr_x, tr_y, te_x, te_y = pickle.load(f)

        tr_x = torch.from_numpy(tr_x).float().unsqueeze(1)
        te_x = torch.from_numpy(te_x).float().unsqueeze(1)
        tr_y = torch.tensor(tr_y, dtype=torch.long)
        te_y = torch.tensor(te_y, dtype=torch.long)

        model = ShallowConvNet(nb_classes=5)
        trainer = TrainerUni(
            model,
            data=[tr_x, tr_y, te_x, te_y],
            epochs=485,
            subject=subject,
            device=device,
        )

        trainer.train()
