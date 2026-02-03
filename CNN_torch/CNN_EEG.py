import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.
    """

    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5,
                 kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
        super(EEGNet, self).__init__()

        self.Chans = Chans
        self.Samples = Samples

        # Block 1: Temporal Convolution + Depthwise Spatial Convolution
        self.block1 = nn.Sequential(
            # Padding='same' requires PyTorch >= 1.9
            nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise Convolution
            nn.Conv2d(F1, D * F1, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D * F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )

        # Block 2: Separable Convolution
        self.block2 = nn.Sequential(
            # Separable Conv Part 1 (Depthwise)
            nn.Conv2d(D * F1, D * F1, (1, 16), padding='same', groups=D * F1, bias=False),
            # Separable Conv Part 2 (Pointwise)
            nn.Conv2d(D * F1, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )

        self.flatten = nn.Flatten()

        # Dynamically calculate the size of the linear layer input
        # This prevents errors if you change Chans or Samples
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, Chans, Samples)
            x = self.block1(dummy_input)
            x = self.block2(x)
            x = self.flatten(x)
            n_flatten = x.shape[1]

        self.classifier = nn.Linear(n_flatten, nb_classes)

    def forward(self, x):
        # Input shape: (Batch, Chans, Samples)
        # We need to add the "Channel" dimension for Conv2d: (Batch, 1, Chans, Samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.block1(x)
        x = self.block2(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x  # Return logits (CrossEntropyLoss handles Softmax)


class EEGNetTrainer:
    """
    Helper class to handle training and validation loops.
    """

    def __init__(self, model, train_dataset, val_dataset, batch_size=32, epochs=100, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.epochs = epochs

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Ensure labels are Long type for CrossEntropy
            if labels.dtype != torch.long:
                labels = labels.long()

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate_epoch(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                if labels.dtype != torch.long:
                    labels = labels.long()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        return val_loss / len(self.test_loader), accuracy

    def train(self):
        print(f"Starting training for {self.epochs} epochs...")
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            val_loss, accuracy = self.validate_epoch()

            # Print every epoch (or change logic to print every N epochs)
            print(f'Epoch {epoch + 1}/{self.epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Acc: {accuracy:.2f}%')

    def predict(self, dataset=None):
        """
        Returns predictions for the internal test loader or a new dataset.
        """
        loader = self.test_loader
        if dataset is not None:
            loader = DataLoader(dataset, batch_size=32, shuffle=False)

        predictions = []
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().tolist())
        return predictions


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Create dummy data (Batch, Channels, Time)
    # 100 samples, 64 channels, 128 time points
    X_train = torch.randn(100, 64, 128)
    y_train = torch.randint(0, 4, (100,))  # 4 classes

    X_val = torch.randn(20, 64, 128)
    y_val = torch.randint(0, 4, (20,))

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    # 2. Initialize Model
    # Note: Samples=128 matches our dummy data time dimension
    model = EEGNet(nb_classes=4, Chans=64, Samples=128, dropoutRate=0.25)

    # 3. Train
    trainer = EEGNetTrainer(model, train_ds, val_ds, batch_size=16, epochs=5)
    trainer.train()