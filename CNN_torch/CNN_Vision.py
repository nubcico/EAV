import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

from sklearn.metrics import f1_score


IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
    ),
])

class VideoModel(nn.Module):
    def __init__(self, num_labels: int = 5, ratio: int = 1):
        super().__init__()
        self.num_labels = num_labels
        self.ratio = ratio

        backbone = resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.attn_fc1 = nn.Linear(2048 // ratio, 2048)
        self.attn_fc2 = nn.Linear(2048, 2048)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_labels),
        )

    def channel_attention(self, x):
        avg_feat = self.avg_pool(x).view(x.size(0), -1)
        max_feat = self.max_pool(x).view(x.size(0), -1)

        avg_feat = self.attn_fc2(self.attn_fc1(avg_feat))
        max_feat = self.attn_fc2(self.attn_fc1(max_feat))

        return avg_feat + max_feat

    def forward(self, x):
        x = self.feature_extractor(x)
        attn = self.channel_attention(x)
        x = x * attn.unsqueeze(-1).unsqueeze(-1)
        x = self.global_pool(x)
        return self.classifier(x)


class ImageClassifierTrainer:
    def __init__(
        self,
        data,
        num_labels=5,
        lr=5e-5,
        batch_size=128,
    ):
        self.tr_x, self.tr_y, self.te_x, self.te_y = data
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.initial_lr = lr

        self.frames_per_sample = self.tr_x.shape[1]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = IMAGE_TRANSFORM
        self.model = VideoModel(num_labels).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        print("Preprocessing images...")
        self.train_loader = self._build_loader(self.tr_x, self.tr_y, shuffle=True)
        self.test_loader = self._build_loader(self.te_x, self.te_y, shuffle=False)
        print("Done.")

    def _build_loader(self, x, y, shuffle=True):
        x_processed = self._preprocess_images(x)
        y_expanded = torch.from_numpy(
            np.repeat(y, self.frames_per_sample)
        ).long()

        dataset = TensorDataset(
            x_processed.view(-1, 3, 224, 224),
            y_expanded,
        )

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _preprocess_images(self, images):
        processed = []
        for frame_set in images:
            for frame in frame_set:
                img = Image.fromarray(frame)
                processed.append(self.transform(img))
        return torch.stack(processed).to(self.device)

    def accuracy(outputs, labels):
        preds = outputs.argmax(dim=1)
        return (preds == labels).float().mean().item()

    def train(self, epochs=3, lr=None, freeze=True):
        lr = lr if lr is not None else self.initial_lr
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        for p in self.model.feature_extractor.parameters():
            p.requires_grad = not freeze

        print(f"Training ({'frozen' if freeze else 'unfrozen'}) | lr={lr}")

        for epoch in range(epochs):
            self.model.train()
            train_acc = 0.0

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()

                train_acc += self.accuracy(out, y)

            train_acc /= len(self.train_loader)


            self.model.eval()
            test_acc = 0.0
            outputs_all = []

            with torch.no_grad():
                for x, y in self.test_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.model(x)
                    test_acc += self.accuracy(out, y)
                    outputs_all.append(out.cpu().numpy())

            test_acc /= len(self.test_loader)

            if epoch == epochs - 1 and not freeze:
                self.outputs_test = np.concatenate(outputs_all, axis=0)

            print(
                f"Epoch {epoch + 1} | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Test Acc: {test_acc * 100:.2f}%"
            )

    def clear_loaders(self):
        del self.train_loader
        del self.test_loader
        torch.cuda.empty_cache()


if __name__ == "__main__":
    test_acc_all = []
    test_f1_all = []

    DATA_DIR = r"C:\Users\minho.lee\Dropbox\Projects\EAV\Feature_vision"

    for subject_id in range(1, 2):
        print(f"\nSubject {subject_id:02d}")

        file_path = os.path.join(
            DATA_DIR,
            f"subject_{subject_id:02d}_vis.pkl",
        )

        with open(file_path, "rb") as f:
            tr_x, tr_y, te_x, te_y = pickle.load(f)

        trainer = ImageClassifierTrainer(
            data=[tr_x, tr_y, te_x, te_y],
            num_labels=5,
            lr=5e-5,
            batch_size=32,
        )

        trainer.train(epochs=3, lr=5e-4, freeze=True)
        trainer.train(epochs=3, lr=5e-6, freeze=False)
        trainer.clear_loaders()

        logits = trainer.outputs_test
        logits = logits.reshape(200, 25, 5).mean(axis=1)

        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == te_y)
        f1 = f1_score(te_y, preds, average="weighted")

        test_acc_all.append(acc)
        test_f1_all.append(f1)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1-score:  {f1:.4f}")

    test_acc_all = np.array(test_acc_all).reshape(-1, 1)
    test_f1_all = np.array(test_f1_all).reshape(-1, 1)
