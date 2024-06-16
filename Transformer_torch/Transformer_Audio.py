import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForAudioClassification
from torch.utils.data import DataLoader, TensorDataset
from transformers import ASTFeatureExtractor
import numpy as np

class AudioModelTrainer:
    def __init__(self, DATA, model_path, sub = '', num_classes=5, weight_decay=1e-5, lr=0.001, batch_size=128):

        self.tr, self.tr_y, self.te, self.te_y = DATA
        self.tr_x = self._feature_extract(self.tr)
        self.te_x = self._feature_extract(self.te)

        self.sub = sub
        self.batch_size = batch_size

        self.train_dataloader = self._prepare_dataloader(self.tr_x, self.tr_y, shuffle=True)
        self.test_dataloader = self._prepare_dataloader(self.te_x, self.te_y, shuffle=False)

        self.model = AutoModelForAudioClassification.from_pretrained(model_path)
        # Modify classifier to fit the number of classes
        self.model.classifier.dense = torch.nn.Linear(self.model.classifier.dense.in_features, num_classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Setup optimizer and loss function
        self.initial_lr = lr
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.initial_lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def _prepare_dataloader(self, x, y, shuffle=False):
        dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def _feature_extract(self, x):
        feature_extractor = ASTFeatureExtractor()
        ft = feature_extractor(x, sampling_rate=16000, padding='max_length',
                               return_tensors='pt')
        return ft['input_values']

    def train(self, epochs=20, lr=None, freeze=True):
        lr = lr if lr is not None else self.initial_lr
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        # Freeze or unfreeze model parameters based on the freeze flag
        for param in self.model.parameters():
            param.requires_grad = not freeze
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        # Wrap the model with DataParallel
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        for epoch in range(epochs):
            self.model.train()
            train_correct, train_total = 0, 0

            total_batches = len(self.train_dataloader)
            for batch_idx, batch in enumerate(self.train_dataloader, start=1):
                #print(f'batch ({batch_idx}/{total_batches})')

                x, t = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()
                logits = self.model(x).logits
                loss = self.loss_fn(logits, t)
                if loss.dim() > 0:
                    loss = loss.mean()
                else:
                    loss = loss
                loss.backward()
                self.optimizer.step()

                train_correct += (logits.argmax(dim=-1) == t).sum().item()
                train_total += t.size(0)
            train_accuracy = train_correct / train_total

            self.model.eval()
            correct, total = 0, 0
            outputs_batch = []
            with torch.no_grad():
                for x, t in self.test_dataloader:
                    x, t = x.to(self.device), t.long().to(self.device)
                    logits = self.model(x).logits
                    correct += (logits.argmax(dim=-1) == t).sum().item()
                    total += t.size(0)

                    logits_cpu = logits.detach().cpu().numpy()
                    outputs_batch.append(logits_cpu)
                test_accuracy = correct / total
            if epoch == epochs-1 and not freeze: # we saved test prediction only at last epoch, and finetuning
                self.outputs_test = np.concatenate(outputs_batch, axis=0)

            print(f"Epoch {epoch + 1}/{epochs}, Training Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%")
            with open('training_performance_audio.txt', 'a') as f:
                f.write(f"{self.sub}, Epoch {epoch + 1}, Test Accuracy: {test_accuracy * 100:.2f}%\n")



