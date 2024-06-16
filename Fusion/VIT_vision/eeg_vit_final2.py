import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
class PositionalEncoding(nn.Module):
    """Positional encoding.
    https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html
    """
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.p = torch.zeros((1, max_len, num_hiddens))
        x = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.p[:, :, 0::2] = torch.sin(x)
        self.p[:, :, 1::2] = torch.cos(x)

    def forward(self, x): # note we carefully add the positional encoding, omitted
        x = x #+ self.p[:, :x.shape[1], :].to(x.device)
        return self.dropout(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
        )

        self.layernorm0 = nn.LayerNorm(embed_dim)
        self.layernorm1 = nn.LayerNorm(embed_dim)

        self.dropout = dropout

    def forward(self, x):
        y, att = self.attention(x, x, x)
        y = F.dropout(y, self.dropout, training=self.training)
        x = self.layernorm0(x + y)
        y = self.mlp(x)
        y = F.dropout(y, self.dropout, training=self.training)
        x = self.layernorm1(x + y)
        return x

class EEGClassificationModel(nn.Module):
    def __init__(self, eeg_channel, dropout=0.1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(
                eeg_channel, eeg_channel, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel),
            nn.ReLU(True),
            nn.Dropout1d(dropout),
            nn.Conv1d(
                eeg_channel, eeg_channel * 2, 11, 1, padding=5, bias=False
            ),
            nn.BatchNorm1d(eeg_channel * 2),
        )

        self.transformer = nn.Sequential(
            PositionalEncoding(eeg_channel * 2, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
            TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
        )

        self.mlp = nn.Sequential(
            nn.Linear(eeg_channel * 2, eeg_channel // 2),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(eeg_channel // 2, 5),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = x.mean(dim=-1)
        x = self.mlp(x)
        return x

class EEGModelTrainer:
    def __init__(self, DATA, model = [], sub = '', lr=0.001, batch_size = 64):
        if model:
            self.model = model
        else:
            self.model = EEGClassificationModel(eeg_channel=30)

        self.tr, self.tr_y, self.te, self.te_y = DATA
        self.batch_size = batch_size
        self.test_acc = float()

        self.train_dataloader = self._prepare_dataloader(self.tr, self.tr_y, shuffle=True)
        self.test_dataloader = self._prepare_dataloader(self.te, self.te_y, shuffle=False)

        self.initial_lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.initial_lr)

        # Automatically use GPU if available, else fallback to CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(self.device)

    def _prepare_dataloader(self, x, y, shuffle=False):
        dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        predictions = []
        accuracies = []

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions.extend(predicted.cpu().numpy())
                accuracies.extend((predicted == labels).cpu().numpy())
        accuracy = correct / total
        print(f'Test Accuracy: {accuracy:.2f}')
        return accuracy, predictions

    def train(self, epochs=25, lr=None, freeze=False):
        lr = lr if lr is not None else self.initial_lr
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        # Freeze or unfreeze model parameters based on the freeze flag
        # we train the eeg model from the scratch
        for param in self.model.parameters():
            param.requires_grad = not freeze

        # Wrap the model with DataParallel
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            print("GPU:", torch.cuda.device_count())

        for epoch in range(epochs):
            # Variables to store performance metrics
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            # Training phase
            self.model.train()
            for inputs, labels in self.train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

            train_loss = running_loss / len(self.train_dataloader.dataset)
            train_accuracy = correct_predictions / total_predictions

            # Validation phase
            self.model.eval()
            running_val_loss = 0.0
            val_correct_predictions = 0
            val_total_predictions = 0
            with torch.no_grad():
                for inputs, labels in self.test_dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    val_loss = self.criterion(outputs, labels)
                    running_val_loss += val_loss.item() * inputs.size(0)

                    _, predicted = torch.max(outputs.data, 1)
                    val_total_predictions += labels.size(0)
                    val_correct_predictions += (predicted == labels).sum().item()

            val_loss = running_val_loss / len(self.test_dataloader.dataset)
            val_accuracy = val_correct_predictions / val_total_predictions

            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        self.test_acc = val_accuracy
        return self.model


if __name__ == "__main__":
    # Toy dataste
   # data = torch.randn(1000, 30, 500)
   # labels = torch.randint(0, 5, (1000,))

    #dataset = TensorDataset(data, labels)

  #  train_size = int(0.5 * len(dataset))
  #  val_size = len(dataset) - train_size
  #  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    import pickle
    import os
    from sklearn.metrics import f1_score

    acc_all = list()
    f1_all = list()
    for sub in range(1, 43):
        file_path = "C:/Users/minho.lee/Dropbox/Datasets/EAV/Input_images/EEG/"
        file_name = f"subject_{sub:02d}_eeg.pkl"
        file_ = os.path.join(file_path, file_name)

        with open(file_, 'rb') as f:
            eeg_list2 = pickle.load(f)
        tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = eeg_list2
        data = [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg]


        # Create DataLoaders for training and validation sets
        batch_size = 32

        model = EEGClassificationModel(eeg_channel=30)  # Example: 64 EEG channels
        trainer = EEGModelTrainer(model = model, DATA = data)
        trainer.train(epochs=100)

        [acc, pred] = trainer.evaluate()
        acc_all.append(acc)

        f1 = f1_score(te_y_eeg, pred, average='weighted')
        f1_all.append(f1)
        print(acc_all)
    a2 = 3

