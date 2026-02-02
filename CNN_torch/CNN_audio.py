import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

class AudioModel(nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.MaxPool1d(8),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv1d(128, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Linear(128 * 22, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def create_dataloader(x, y, batch_size=64, shuffle=True):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class ActivationSaver:
    def __init__(self, model, val_loader, save_dir):
        self.model = model
        self.val_loader = val_loader
        self.save_dir = save_dir
        self.epoch = 0

        os.makedirs(save_dir, exist_ok=True)

    def save(self):
        self.model.eval()
        activations = []

        with torch.no_grad():
            for x, _ in self.val_loader:
                x = x.permute(0, 2, 1)  # (B, 1, T)
                out = self.model(x)
                activations.append(out.cpu().numpy())

        activations = np.concatenate(activations, axis=0)
        path = os.path.join(self.save_dir, f"activations_epoch_{self.epoch + 1}.pth")
        torch.save(activations, path)

        self.epoch += 1
        self.model.train()


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-3,
    save_dir="activations",
    subject_id=None,
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    activation_saver = ActivationSaver(model, val_loader, save_dir)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct, total = 0, 0

        for x, y in train_loader:
            x = x.permute(0, 2, 1).to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        train_acc = 100 * correct / total

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.permute(0, 2, 1).to(device)
                y = y.to(device)

                out = model(x)
                correct += (out.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        val_acc = 100 * correct / total

        activation_saver.save()

        print(
            f"Epoch [{epoch + 1}/{epochs}] | "
            f"Loss: {train_loss / len(train_loader):.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if epoch == epochs - 1 and subject_id is not None:
            model_path = os.path.join(
                r"D:\.spyder-py3\finetuned_cnn_7030",
                f"audio_finetuned_{subject_id}.pth",
            )
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")


acc = []
fscore = []
confusion = []

ROOT_FOLDER = r"D:\EAV"
EMOTIONS = ["Anger", "Neutral", "Sadness", "Calmness", "Happiness"]

NUM_SUBJECTS = 42
TRAIN_SAMPLES_PER_CLASS = 56
TOTAL_SAMPLES_PER_CLASS = 80


for subject_id in range(1, NUM_SUBJECTS + 1):

    print(f"\nProcessing Subject {subject_id:02d}")

    subject_folder = f"subject{subject_id}"
    subject_path = os.path.join(ROOT_FOLDER, subject_folder)
    audio_folder = os.path.join(subject_path, "Audio")

    if not os.path.exists(audio_folder):
        print(f"Audio folder not found: {audio_folder}")
        continue

    categorized_files = {emotion: [] for emotion in EMOTIONS}

    for filename in sorted(os.listdir(audio_folder)):
        if not filename.endswith(".wav"):
            continue

        for emotion in EMOTIONS:
            if emotion in filename:
                # Skip incorrect Neutral overlaps
                if emotion != "Neutral" and "Neutral" in filename:
                    continue

                categorized_files[emotion].append(
                    os.path.join(audio_folder, filename)
                )
    data, labels = [], []

    for emotion in EMOTIONS:
        for audio_file in categorized_files[emotion]:
            x, y = proc_data(audio_file, emotion)
            data.extend(x)
            labels.extend(y)
    x_train, y_train = [], []
    x_test, y_test = [], []

    for i in range(len(EMOTIONS)):
        start = i * TOTAL_SAMPLES_PER_CLASS
        mid = start + TRAIN_SAMPLES_PER_CLASS
        end = start + TOTAL_SAMPLES_PER_CLASS

        x_train.extend(data[start:mid])
        y_train.extend(labels[start:mid])

        x_test.extend(data[mid:end])
        y_test.extend(labels[mid:end])
      
    x_train = np.expand_dims(np.array(x_train), axis=2)
    x_test = np.expand_dims(np.array(x_test), axis=2)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    train_loader = create_dataloader(x_train, y_train, batch_size=64)
    val_loader = create_dataloader(x_test, y_test, batch_size=64)

    model = AudioModel(num_classes=len(EMOTIONS))

    train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        num_epochs=100,
        sub=subject_id,
    )

