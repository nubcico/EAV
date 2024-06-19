from Dataload_eeg import *
import torch
import torch.nn as nn
from Fusion.VIT_audio.Transformer_audio import Trainer_uni
#from tensorflow.keras.models import Model
from torch.nn.utils import weight_norm

import os
import pickle 
from torch.utils.data import DataLoader, TensorDataset

# Define the EEGNet model
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
import torch.optim as optim

class EEGNet_tor(nn.Module):
    def __init__(self, nb_classes, Chans=30, Samples=500, dropoutRate=0.5, kernLength=300, F1=8, D=8, F2=64,
                 norm_rate=1.0, dropoutType='Dropout'):
        super(EEGNet_tor, self).__init__()

        # Configure dropout
        self.dropout = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Block 1
        self.firstConv = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.firstBN = nn.BatchNorm2d(F1)
        self.elu = nn.ELU()

        #self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, padding=0, bias=False)
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, padding='valid', bias=False)
        self.depthwiseBN = nn.BatchNorm2d(F1 * D)
        self.depthwisePool = nn.AvgPool2d((1, 4))

        # Applying max-norm constraint
        #self.depthwiseConv.register_forward_hook(
           # lambda module, inputs, outputs: module.weight.data.renorm_(p=2, dim=0, maxnorm=norm_rate))

        # Block 2
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.separableBN = nn.BatchNorm2d(F2)
        self.separablePool = nn.AvgPool2d((1, 8))

        # Final layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * ((Samples // 4 // 8)), nb_classes)
        self.softmax = nn.Softmax(dim=1)

        # Applying max-norm constraint
        #self.dense.register_forward_hook(
            #lambda module, inputs, outputs: module.weight.data.renorm_(p=2, dim=0, maxnorm=norm_rate))

    def forward(self, x):
        x = self.firstConv(x)
        x = self.firstBN(x)
        x = self.elu(x)
        x = self.depthwiseConv(x)
        x = self.depthwiseBN(x)
        x = self.elu(x)
        x = self.depthwisePool(x)
        x = self.dropout(x)
        x = self.separableConv(x)
        x = self.separableBN(x)
        x = self.elu(x)
        x = self.separablePool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x

class Trainer_uni:
    def __init__(self, model, data, lr=1e-4, batch_size=32, num_epochs=10, device=None):

        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        self.tr_x, self.tr_y, self.te_x, self.te_y = data
        self.train_dataloader = self._prepare_dataloader(self.tr_x, self.tr_y, shuffle=True)
        self.test_dataloader = self._prepare_dataloader(self.te_x, self.te_y, shuffle=False)

        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    def _prepare_dataloader(self, x, y, shuffle=False):
        dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def train(self):
        self.model.train()  # Set model to training mode
        for epoch in range(self.num_epochs):
            for batch_idx, (data, targets) in enumerate(self.train_dataloader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                scores = self.model(data)
                loss = self.criterion(scores, targets)

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self.num_epochs}], Step [{batch_idx}/{len(self.train_dataloader)}], Loss: {loss.item():.4f}")

            if self.test_dataloader:
                self.validate()

    def validate(self):
        self.model.eval()  # Set model to evaluation mode
        total_loss = 0
        total_correct = 0
        with torch.no_grad():
            for data, targets in self.test_dataloader:
                data = data.to(self.device)
                targets = targets.to(self.device)

                scores = self.model(data)
                loss = self.criterion(scores, targets)
                total_loss += loss.item()
                predictions = scores.argmax(dim=1)
                total_correct += (predictions == targets).sum().item()

        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = total_correct / len(self.test_dataloader.dataset)
        print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    
    result_acc = list()
    file_path = "C:/Users/minho.lee/Dropbox/Datasets/EAV/Input_images/EEG/"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for i in range(1, 43):
        file_name = f"subject_{i:02d}_eeg.pkl"
        file_ = os.path.join(file_path, file_name)
        print(i)
                
        # Check if the file exists and is not empty
        if not os.path.isfile(file_):
            print(f"File {file_} does not exist.")
            continue
        if os.path.getsize(file_) == 0:
            print(f"File {file_} is empty.")
            continue

        try:
            with open(file_, 'rb') as f:
                eeg_list = pickle.load(f)
        except EOFError as e:
            print(f"Error loading {file_}: {e}")
            continue
        
        tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = eeg_list
        
        #to torch and reshape        
        tr_x_eeg = torch.from_numpy(tr_x_eeg).float().unsqueeze(1).to(device)  # Reshape to (batch, 1, chans, samples)
        tr_y_eeg = torch.tensor(tr_y_eeg, dtype=torch.long).to(device)
        te_x_eeg = torch.from_numpy(te_x_eeg).float().unsqueeze(1).to(device)  # Reshape to (batch, 1, chans, samples)   
        te_y_eeg = torch.tensor(te_y_eeg, dtype=torch.long).to(device)

        # Create DataLoader
        train_dataset = TensorDataset(tr_x_eeg, tr_y_eeg)
        test_dataset = TensorDataset(te_x_eeg, te_y_eeg)
        
        # Parameters
        num_epochs = 350
        norm_rate = 1.0
        batch_size = 32
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = EEGNet_tor(nb_classes=5, Chans=30, Samples=500)
        model.to(device)
        
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training 
        for epoch in range(num_epochs):
            model.train()
            for batch_data, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                # Apply max-norm constraint to tensors
                # (here is max_norm is done instead of in model architecture)
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if 'weight' in name and param.dim() >= 2:
                            param.data = param.data.renorm_(p=2, dim=0, maxnorm=norm_rate)
            
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        
        # Evaluation 
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_labels in test_loader:
                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        acc = correct / total
        result_acc.append(acc)
        
        print(f'Accuracy: {100 * correct / total:.2f}%')
