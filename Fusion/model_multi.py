from Fusion.VIT_audio.Transformer_audio import ViT_Encoder, ast_feature_extract

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from Dataload_audio import DataLoadAudio
from Dataload_eeg import DataLoadEEG
from EAV_datasplit import EAVDataSplit


class MultiModalViT(nn.Module):
    def __init__(self, audio_model, eeg_model):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.audio_model = audio_model
        self.eeg_model = eeg_model
        self.audio_model = audio_model.to(device)
        self.eeg_model = eeg_model.to(device)

        self.classifier = nn.Linear(audio_model.embed_dim + eeg_model.embed_dim, audio_model.num_classes)  # Assume same output features from both models

    def forward(self, audio_x, eeg_x):
        audio_features = self.audio_model.feature(audio_x)[:, 0]  # Extract only class token features
        eeg_features = self.eeg_model.feature(eeg_x)[:, 0]  # Extract only class token features
        combined_features = torch.cat((audio_features, eeg_features), dim=1)
        output = self.classifier(combined_features)
        return output

class TrainerMultiModal:
    def __init__(self, multimodal_model, data_audio, data_eeg, lr=1e-4, batch_size=32, num_epochs=10, sub = ''):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multimodal_model = multimodal_model.to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.sub = sub


        # Prepare data loaders for both audio and EEG
        self.train_dataloader = self._prepare_dataloader(data_audio[0], data_audio[1], data_eeg[0], data_eeg[1],
                                                         shuffle=True)
        self.test_dataloader = self._prepare_dataloader(data_audio[2], data_audio[3], data_eeg[2], data_eeg[3],
                                                        shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.multimodal_model.parameters(), lr=self.lr)

    def _prepare_dataloader(self, x_audio, y_audio, x_eeg, y_eeg, shuffle=False):
        # Ensure audio and EEG data are tensors
        x_audio = torch.tensor(x_audio, dtype=torch.float32)
        y_audio = torch.tensor(y_audio, dtype=torch.long)
        x_eeg = torch.tensor(x_eeg, dtype=torch.float32)
        y_eeg = torch.tensor(y_eeg, dtype=torch.long)

        # Create datasets
        audio_dataset = TensorDataset(x_audio, y_audio)
        eeg_dataset = TensorDataset(x_eeg, y_eeg)

        # Create combined dataset with zip, allowing paired data handling
        combined_dataset = [(a, e) for a, e in zip(audio_dataset, eeg_dataset)]
        dataloader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader

    def train(self):
        self.multimodal_model.train()  # Set model to training mode
        total_loss = 0
        for epoch in range(self.num_epochs):
            for (audio_data, eeg_data) in self.train_dataloader:

                audio_inputs, audio_labels = audio_data
                eeg_inputs, eeg_labels = eeg_data

                # Ensure labels match for both modalities
                assert torch.equal(audio_labels, eeg_labels), "Labels do not match between modalities."

                # Move data to device
                audio_inputs, audio_labels = audio_inputs.to(self.device), audio_labels.to(self.device)
                eeg_inputs, eeg_labels = eeg_inputs.to(self.device), eeg_labels.to(self.device)

                # Forward pass through each model individually for additional losses
                audio_output = self.multimodal_model.audio_model(audio_inputs)
                eeg_output = self.multimodal_model.eeg_model(eeg_inputs)

                # Calculate additional losses
                audio_loss = self.criterion(audio_output, audio_labels)
                eeg_loss = self.criterion(eeg_output, eeg_labels)

                # Combined model forward pass
                scores = self.multimodal_model(audio_inputs, eeg_inputs)
                combined_loss = self.criterion(scores, audio_labels)

                # Total loss
                loss = audio_loss + eeg_loss + combined_loss
                total_loss += loss.item()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()

                #torch.nn.utils.clip_grad_norm_(self.multimodal_model.audio_model.parameters(), max_norm=1.0)
                #torch.nn.utils.clip_grad_norm_(self.multimodal_model.eeg_model.parameters(), max_norm=1.0)
                
                self.optimizer.step()

            print(f"{self.sub}_Epoch {epoch + 1}, Combined Loss: {combined_loss.item():.4f}, Audio Loss: {audio_loss.item():.4f}, EEG Loss: {eeg_loss.item():.4f}")

            # Optionally add validation here
            self.validate()

    def validate(self):
        self.multimodal_model.eval()  # Set the model to evaluation mode
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for (audio_data, eeg_data) in self.test_dataloader:
                audio_inputs, audio_labels = audio_data
                eeg_inputs, eeg_labels = eeg_data

                # Ensure labels match for both modalities (this check is optional but can prevent data mismatches)
                assert torch.equal(audio_labels, eeg_labels), "Mismatch between audio and EEG labels."

                # Move data to the device
                audio_inputs, audio_labels = audio_inputs.to(self.device), audio_labels.to(self.device)
                eeg_inputs, eeg_labels = eeg_inputs.to(self.device), eeg_labels.to(self.device)

                # Forward pass to get output/logits
                scores = self.multimodal_model(audio_inputs, eeg_inputs)

                # Calculate the batch loss
                loss = self.criterion(scores, audio_labels)
                total_loss += loss.item()

                # Convert scores to actual predictions
                _, predicted_labels = torch.max(scores, 1)
                correct_predictions += (predicted_labels == audio_labels).sum().item()
                total_samples += audio_labels.size(0)

        # Calculate average loss and accuracy
        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = (correct_predictions / total_samples) * 100

        with open('aud_eeg_results_0414.txt', 'a') as f:
            f.write(f"{self.sub}_Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n")
        print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")


for i in range(42):
    sub = i+1
    model_aud = ViT_Encoder(classifier = True, img_size=[1024, 128], in_chans=1, patch_size = (16, 16), stride = 10, embed_pos = True)
    model_eeg = ViT_Encoder(classifier = True, img_size=[60, 500], in_chans=1,
                        patch_size = (60, 1), stride = 1, depth=2, num_heads=4,
                        embed_eeg = True, embed_pos = False)
    combined_model = MultiModalViT(model_aud, model_eeg)

    path = r'D:\\Dropbox\\DATASETS\\EAV'
    #path = r'C:\\Users\\minho.lee\\Dropbox\\EAV'
    aud_loader = DataLoadAudio(subject=sub, parent_directory=path)
    [data_aud , data_aud_y] = aud_loader.process()
    division_aud = EAVDataSplit(data_aud, data_aud_y)
    [tr_x_aud, tr_y_aud, te_x_aud , te_y_aud] = division_aud.get_split()
    tr_x_aud_ft = ast_feature_extract(tr_x_aud)
    te_x_aud_ft = ast_feature_extract(te_x_aud)
    data_AUD = [tr_x_aud_ft.unsqueeze(1), tr_y_aud, te_x_aud_ft.unsqueeze(1), te_y_aud]


    eeg_loader = DataLoadEEG(subject=sub, band=[0.5, 45], fs_orig=500, fs_target=100,
                                 parent_directory=path)
    data_eeg, data_eeg_y = eeg_loader.data_prepare()
    division_eeg = EAVDataSplit(data_eeg, data_eeg_y)
    [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg] = division_eeg.get_split()
    data_EEG = [torch.from_numpy(tr_x_eeg).float(), tr_y_eeg, torch.from_numpy(te_x_eeg).float(), te_y_eeg]


    trainer = TrainerMultiModal(multimodal_model = combined_model, data_audio = data_AUD,
                                data_eeg = data_EEG, batch_size=8, num_epochs = 30, sub = sub)

    trainer.train()


'''
############################################### audio
model = ViT_Encoder(classifier = True, img_size=[1024, 128], in_chans=1, patch_size = (16, 16), stride = 16, embed_pos = False)


model_pre = AutoModelForImageClassification.from_pretrained(model_path)


aud_loader = DataLoadAudio(subject=2, parent_directory=r'C:\\Users\\minho.lee\\Dropbox\\EAV')
[data_aud , data_aud_y] = aud_loader.process()
division_aud = EAVDataSplit(data_aud, data_aud_y)
[tr_x_aud, tr_y_aud, te_x_aud , te_y_aud] = division_aud.get_split()
tr_x_aud_ft = ast_feature_extract(tr_x_aud)
te_x_aud_ft = ast_feature_extract(te_x_aud)
data = [tr_x_aud_ft.unsqueeze(1), tr_y_aud, te_x_aud_ft.unsqueeze(1), te_y_aud]

trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=8, num_epochs=10)
trainer.train()


######################## # EEG
from Dataload_eeg import DataLoadEEG


eeg_loader = DataLoadEEG(subject=2, band=[0.5, 45], fs_orig=500, fs_target=100,
                             parent_directory=r'C:\\Users\\minho.lee\\Dropbox\\EAV')
data_eeg, data_eeg_y = eeg_loader.data_prepare()

division_eeg = EAVDataSplit(data_eeg, data_eeg_y)
[tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg] = division_eeg.get_split()
data = [torch.from_numpy(tr_x_eeg).float(), tr_y_eeg, torch.from_numpy(te_x_eeg).float(), te_y_eeg]

model = ViT_Encoder(classifier = True, img_size=[30, 500], in_chans=1,
                    patch_size = (60, 1), stride = 1, depth=4, num_heads=4,
                    embed_eeg = True, embed_pos = False)

trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=32, num_epochs=10)
trainer.train()

model.to(torch.device("cuda"))
ft_x = list()
for i in range(200):
    a = torch.from_numpy(te_x_eeg[i]).float().unsqueeze(0).to(torch.device("cuda"))
    ft_x.append(model.feature(a).cpu())
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

features = torch.stack(ft_x).squeeze()  # Remove extra dimensions, if any
# Extract the first row from each sample
features_first_row = features[:, 0, :].detach()  # This slices out the first row for all samples and detaches it

# Convert labels to a NumPy array for plotting
labels = np.array(te_y_eeg)

# Define the emotion_to_index mapping
emotion_to_index = {
    'Neutral': 0,
    'Happiness': 3,
    'Sadness': 1,
    'Anger': 2,
    'Calmness': 4
}

# Reverse mapping to get index to emotion mapping for plotting
index_to_emotion = {v: k for k, v in emotion_to_index.items()}

# Use t-SNE to reduce the dimensionality
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features_first_row)

# Plotting
plt.figure(figsize=(10, 8))
for i, emotion in index_to_emotion.items():
    indices = labels == i
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=emotion)
plt.legend()
plt.title('t-SNE Visualization of the First Row of Features')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()

'''