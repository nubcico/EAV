import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class Audio(nn.Module):
    def __init__(self, input_length=180, num_classes=5):
        super(AudioNet, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 256, kernel_size=5, padding='same')
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, padding='same', padding_mode='replicate')
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, padding='same', padding_mode='replicate')
        self.conv4 = nn.Conv1d(128, 128, kernel_size=5, padding='same', padding_mode='replicate')
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool1d(8)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128 * (input_length // 8), num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class AudioTrainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size=32, epochs=100):
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters())
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("Training on:", self.device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                preds = output.argmax(dim=1, keepdim=True)
                correct += preds.eq(target.view_as(preds)).sum().item()
        accuracy = 100. * correct / len(self.val_loader.dataset)
        return total_loss / len(self.val_loader), accuracy

    def train(self):
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss, val_accuracy = self.validate_epoch()
            print(f'Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Accuracy {val_accuracy:.2f}%')
