from Transformer_video import ViT_Encoder_Video
from Transformer import ViT_Encoder_Audio, ast_feature_extract
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from Dataload_audio import DataLoadAudio
from EAV_datasplit import EAVDataSplit
import numpy as np
from transformers import AutoImageProcessor
from torch.cuda.amp import autocast
from timm.layers import Mlp, DropPath, use_fused_attn
import torch.nn.functional as F

class Attention(nn.Module):
    #fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,  # should be true
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        #q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class MultiModalViT(nn.Module):
    def __init__(self, audio_model, video_model):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim=768
        num_heads=12
        qkv_bias=True
        attn_drop=0.0
        proj_drop=0.0
        self.audio_model = audio_model.to(device)
        self.video_model = video_model.to(device)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        mlp_layer: nn.Module = Mlp
        act_layer: nn.Module = nn.GELU
        norm_layer: nn.Module = nn.LayerNorm
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * 4),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.norm2 = norm_layer(dim)
        self.classifier = nn.Linear(audio_model.embed_dim + video_model.embed_dim, audio_model.num_classes)  # Assume same output features from both models

    def forward(self, audio_x, video_x):
        audio_features = self.audio_model.feature(audio_x)  # Extract only class token features
        #print(audio_features.shape)
        batch_size, num_samples, c, h, w = video_x.size()
        video_x = video_x.view(batch_size * num_samples, c, h, w)  # Reshape to (batch_size * 25, 3, 224, 224)
        video_features = self.video_model.feature(video_x)  # Extract class token features for all samples
        #print(video_features)
        video_features = video_features.view(batch_size, num_samples, -1, 768)  # Reshape to (batch_size, 25, embed_dim)
        #print(video_features)
        video_features = video_features.mean(dim=1)  # Average the 25 class tokens
       #print(video_features.shape)
        
        
        x = torch.cat((audio_features, video_features), dim=1)
        x = x + self.attn(x)
        x = x + self.mlp(self.norm2(x))
        #print(combined_features.shape)
        audio_features=x[:,0]
        video_features=x[:,1213]
        combined_features=torch.cat((audio_features, video_features), dim=1)
        output = self.classifier(combined_features)
        return output


class TrainerMultiModal:
    def __init__(self, multimodal_model, data_audio, data_video, lr=1e-4, batch_size=32, num_epochs=10, sub=''):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multimodal_model = multimodal_model.to(self.device)
        self.initial_lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.sub = sub

        # Prepare data loaders for both audio and video
        self.train_dataloader = self._prepare_dataloader(data_audio[0], data_audio[1], data_video[0], data_video[1], shuffle=True)
        self.test_dataloader = self._prepare_dataloader(data_audio[2], data_audio[3], data_video[2], data_video[3], shuffle=False)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.multimodal_model.parameters(), lr=self.initial_lr)
    
    def preprocess_images(self, image_list):
        pixel_values_list = []
        processor = AutoImageProcessor.from_pretrained(r'C:\Users\user.DESKTOP-HI4HHBR\Downloads\facial_emotions_image_detection (1)')
        for img_set in image_list:
            for img in img_set:
                processed = processor(images=img, return_tensors="pt")
                pixel_values = processed.pixel_values.squeeze()
                pixel_values_list.append(pixel_values)
        return torch.stack(pixel_values_list)
    
    def _prepare_dataloader(self, x_audio, y_audio, x_video, y_video, shuffle=False):
        x_video = self.preprocess_images(x_video).view(-1, 25, 3, 224, 224)  # Adjust to have 25 samples
        y_video = torch.from_numpy(y_video).long()
        x_audio = torch.tensor(x_audio, dtype=torch.float32)
        y_audio = torch.from_numpy(y_audio).long()
        
        audio_dataset = TensorDataset(x_audio, y_audio)
        video_dataset = TensorDataset(x_video, y_video)

        combined_dataset = [(a, e) for a, e in zip(audio_dataset, video_dataset)]
        dataloader = DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=shuffle)
        
        return dataloader

    def train(self, freeze, epochs=20, lr=None):
        total_correct = 0
        total_samples = 0
        total_loss = 0
        lr = lr if lr is not None else self.initial_lr
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                
        for param in self.multimodal_model.parameters():
            param.requires_grad = not freeze
        for param in self.multimodal_model.classifier.parameters():
            param.requires_grad = True
        total_loss = 0
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.multimodal_model = nn.DataParallel(self.multimodal_model)
            
        if isinstance(self.multimodal_model, nn.DataParallel):
            self.multimodal_model = self.multimodal_model.module
        for epoch in range(epochs):
            self.multimodal_model.train()
            for i, (audio_data, video_data) in enumerate(self.train_dataloader):
                audio_inputs, audio_labels = audio_data
                video_inputs, video_labels = video_data

                # Ensure labels match for both modalities
                assert torch.equal(audio_labels, video_labels), "Labels do not match between modalities."

                # Move data to device
                audio_inputs, audio_labels = audio_inputs.to(self.device), audio_labels.to(self.device)
                video_inputs, video_labels = video_inputs.to(self.device), video_labels.to(self.device)

                # Clear GPU memory
                torch.cuda.empty_cache()

                with autocast():  # Mixed precision training
                    audio_loss = 0
                    video_loss = 0

                    torch.cuda.empty_cache()

                    # Combined model forward pass
                    scores = self.multimodal_model(audio_inputs, video_inputs)
                    combined_loss = self.criterion(scores, audio_labels)

                    # Total loss
                    loss = audio_loss + video_loss + combined_loss
                    total_loss += loss.item()

                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                # Accuracy calculation
                _, predicted_labels = torch.max(scores, 1)
                total_correct += (predicted_labels == audio_labels).sum().item()
                total_samples += audio_labels.size(0)

                
            training_accuracy = total_correct / total_samples
            print(f"{self.sub}_Epoch {epoch + 1}, Training Accuracy: {training_accuracy:.4f}")
            accuracy = self.validate()
            print(f"Epoch {epoch + 1}, Validation Accuracy: {accuracy:.2f}")
            with open('aud_video_results_attention.txt', 'a') as f:
                f.write(f"Subject {self.sub} Epoch {epoch + 1} Testing Accuracy: {accuracy}\n")

    def validate(self):
        self.multimodal_model.eval()
        correct_predictions = 0
        total_samples = 0
        total_loss = 0
        with torch.no_grad():
            for (audio_data, video_data) in self.test_dataloader:
                audio_inputs, audio_labels = audio_data
                video_inputs, video_labels = video_data

                # Ensure labels match for both modalities (this check is optional but can prevent data mismatches)
                assert torch.equal(audio_labels, video_labels), "Mismatch between audio and video labels."

                # Move data to the device
                audio_inputs, audio_labels = audio_inputs.to(self.device), audio_labels.to(self.device)
                video_inputs, video_labels = video_inputs.to(self.device), video_labels.to(self.device)

                # Forward pass to get output/logits
                scores = self.multimodal_model(audio_inputs, video_inputs)

                # Calculate the batch loss
                loss = self.criterion(scores, audio_labels)
                total_loss += loss.item()

                # Convert scores to actual predictions
                _, predicted_labels = torch.max(scores, 1)
                correct_predictions += (predicted_labels == audio_labels).sum().item()
                total_samples += audio_labels.size(0)

        # Calculate average loss and accuracy
        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = (correct_predictions / total_samples)

        #print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy


import os
import pickle 
str=[2,4,6,17,31]
str=[4]
for i in range(1,43):
    sub=i
    model_aud = ViT_Encoder_Audio(classifier=True, img_size=[1024, 128], in_chans=1, patch_size=(16, 16), stride=10, embed_pos=True)
    model_path = f"model_with_weights_audio_finetuned_{i}.pth"
    model_aud.load_state_dict(torch.load(model_path))
    
    model_vid = ViT_Encoder_Video(classifier=True, img_size=(224, 224), in_chans=3, patch_size=(16, 16), stride=16, embed_pos=True)
    model_path = f"model_with_weights_video_finetuned_{i}.pth"
    model_vid.load_state_dict(torch.load(model_path))
    
    combined_model = MultiModalViT(model_aud, model_vid)

    path = r'D:\\EAV'
    aud_loader = DataLoadAudio(subject=sub, parent_directory=path)
    [data_aud, data_aud_y] = aud_loader.process()
    division_aud = EAVDataSplit(data_aud, data_aud_y)
    [tr_x_aud, tr_y_aud, te_x_aud, te_y_aud] = division_aud.get_split()
    tr_x_aud_ft = ast_feature_extract(tr_x_aud)
    te_x_aud_ft = ast_feature_extract(te_x_aud)
    data_AUD = [tr_x_aud_ft.unsqueeze(1), tr_y_aud, te_x_aud_ft.unsqueeze(1), te_y_aud]

    file_name = f"subject_{i:02d}_vis.pkl"
    file_ = os.path.join(r"C:\Users\user.DESKTOP-HI4HHBR\Downloads\Feature_vision", file_name)

    with open(file_, 'rb') as f:
        vis_list2 = pickle.load(f)
    tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = vis_list2
    data_video = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]

    trainer = TrainerMultiModal(multimodal_model=combined_model, data_audio=data_AUD, data_video=data_video, batch_size=8, num_epochs=30, sub=sub)

    trainer.train(freeze=True, epochs=20, lr=5e-4)
    #trainer.train(freeze=False, epochs=10, lr=5e-6)
