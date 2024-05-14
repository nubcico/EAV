import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from transformers import ASTFeatureExtractor

import timm
from timm.models import vision_transformer
import torch.nn.functional as F
from timm.layers import Mlp, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType

from Dataload_audio import DataLoadAudio
from EAV_datasplit import EAVDataSplit
import numpy as np
import os

class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.

        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Attention(nn.Module):
    #fused_attn: Final[bool]
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = True,
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
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
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
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,

            #init_values: Optional[float] = None,
            init_values=None,  # mhlee
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = CustomLayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = CustomLayerNorm(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x= x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x= x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x

class CustomLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        # Call the constructor of nn.LayerNorm with the same arguments
        super(CustomLayerNorm, self).__init__(*args, **kwargs)
        
        # Override the default value of epsilon (eps)
        self.eps = 1e-12 # it was 1e-12 in the ast model

from itertools import repeat
import collections
from enum import Enum

class EEG_decoder(nn.Module):
    def __init__(self, eeg_channel = 30, dropout=0.1):
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
    def forward(self, x):
        x = self.conv(x)
        return x

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    dynamic_img_pad: torch.jit.Final[bool]
    def __init__(
            self,
            img_size = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            stride = None,
            embed_dim: int = 768,
            norm_layer = None,
            flatten: bool = True,
            output_fmt = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        self.patch_size = tuple(patch_size)

        if img_size is not None:
            self.img_size = tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

            # flatten spatial dim and transpose to channels last, kept for bwd compat
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        # updated_mh
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=16, stride=10)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        x=x.permute(0,1,3,2) # needs to be permuted according to the outputs from the ast
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x

#emb = PatchEmbed(img_size=[1024, 128], in_chans=1)
#out = emb(torch.randn(1, 1, 1024, 128))

#emb = PatchEmbed(img_size=[30, 500], in_chans=1, patch_size = (1, 100))
#out = emb(torch.randn(200, 1, 30, 500))

#aaa = EEG_decoder()
#out = aaa(torch.randn(1, 30, 500))
#out2 = out.unsqueeze(1)
#emb = PatchEmbed(img_size=[60, 500], in_chans=1, patch_size = (60, 1))
#out3 = emb(out2)


class ViT_Encoder(nn.Module):
    def __init__(self, img_size=[224, 224], in_chans = 3, patch_size=16, stride = 16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 classifier : bool = False, num_classes = 527, embed_eeg = False, embed_pos = True):
        super().__init__()
        # updated_mh
        #self.num_patches = (img_size // patch_size) ** 2
        self.embed_eeg = embed_eeg
        self.embed_pos = embed_pos

        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.stride = stride
        self.eeg_embed = EEG_decoder()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride = stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1214, embed_dim)) #hardcoded dimension "self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 2, config.hidden_size))"
        self.pos_drop = nn.Dropout(p=0.0, inplace=False)
        self.feature_map = None  # this will contain the ViT feature map (including CLASS token)
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = CustomLayerNorm(embed_dim)
        self.norm_cls = CustomLayerNorm(embed_dim)
        if classifier:
            self.head = nn.Linear(embed_dim, num_classes, bias=True)
        else:
            self.head = []


    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # 복제
        distillation_tokens =self.distillation_token.expand(B, -1, -1) # was used in ast along with cls_token
        x = torch.cat((cls_tokens,distillation_tokens, x), dim=1)        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
            
        self.feature_map = x
        if self.head:  # classifier mode
            x = self.norm_cls(x)
            x = self.head(x[:, 0])
        return x


class Trainer_uni:
    def __init__(self, model, data, lr=1e-4, batch_size=32, num_epochs=10, device=None):

        self.initial_lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.tr_x, self.tr_y, self.te_x, self.te_y = data
        self.train_dataloader = self._prepare_dataloader(self.tr_x, self.tr_y, shuffle=True)
        self.test_dataloader = self._prepare_dataloader(self.te_x, self.te_y, shuffle=False)
        self.model=model
        
        ######################################################################
        ###   Starting the assignment of weights and biases to the model   ###
        ######################################################################
        
        
        filename = "D:/weights/audio_spectrogram_transformer.embeddings.patch_embeddings.projection.weight.pth"
        weight_tensor = torch.load(filename)
        self.model.patch_embed.proj.weight.data = weight_tensor
        filename = "D:/biases/audio_spectrogram_transformer.embeddings.patch_embeddings.projection.bias.pth"
        bias_tensor=torch.load(filename)
        self.model.patch_embed.proj.bias.data = bias_tensor
        
       
        for idx in range (0,12):
        
            filename = f"D:/weights/audio_spectrogram_transformer.encoder.layer.{idx}.layernorm_before.weight.pth"
            weight_tensor = torch.load(filename)
            self.model.blocks[idx].norm1.weight.data = weight_tensor
            filename = f"D:/biases/audio_spectrogram_transformer.encoder.layer.{idx}.layernorm_before.bias.pth"
            bias_tensor = torch.load(filename)
            self.model.blocks[idx].norm1.bias.data = bias_tensor
            
            
            filename1 = f"D:/weights/audio_spectrogram_transformer.encoder.layer.{idx}.attention.attention.query.weight.pth"
            filename2 = f"D:/weights/audio_spectrogram_transformer.encoder.layer.{idx}.attention.attention.key.weight.pth"
            filename3 = f"D:/weights/audio_spectrogram_transformer.encoder.layer.{idx}.attention.attention.value.weight.pth"
            weight_tensor1 = torch.load(filename1)
            weight_tensor2 = torch.load(filename2)
            weight_tensor3 = torch.load(filename3)
            weight_tensor= torch.cat((weight_tensor1, weight_tensor2, weight_tensor3), dim=0)
            self.model.blocks[idx].attn.qkv.weight.data = weight_tensor
            filename1 = f"D:/biases/audio_spectrogram_transformer.encoder.layer.{idx}.attention.attention.query.bias.pth"
            filename2 = f"D:/biases/audio_spectrogram_transformer.encoder.layer.{idx}.attention.attention.key.bias.pth"
            filename3 = f"D:/biases/audio_spectrogram_transformer.encoder.layer.{idx}.attention.attention.value.bias.pth"
            bias_tensor1 = torch.load(filename1)
            bias_tensor2 = torch.load(filename2)
            bias_tensor3 = torch.load(filename3)
            bias_tensor= torch.cat((bias_tensor1, bias_tensor2, bias_tensor3), dim=0)
            self.model.blocks[idx].attn.qkv.bias.data = bias_tensor
            
            
            filename = f"D:/weights/audio_spectrogram_transformer.encoder.layer.{idx}.attention.output.dense.weight.pth"
            weight_tensor = torch.load(filename)
            self.model.blocks[idx].attn.proj.weight.data = weight_tensor
            filename = f"D:/biases/audio_spectrogram_transformer.encoder.layer.{idx}.attention.output.dense.bias.pth"
            bias_tensor = torch.load(filename)
            self.model.blocks[idx].attn.proj.bias.data = bias_tensor
            
            filename = f"D:/weights/audio_spectrogram_transformer.encoder.layer.{idx}.layernorm_after.weight.pth"
            weight_tensor = torch.load(filename)
            self.model.blocks[idx].norm2.weight.data = weight_tensor
            filename = f"D:/biases/audio_spectrogram_transformer.encoder.layer.{idx}.layernorm_after.bias.pth"
            bias_tensor = torch.load(filename)
            self.model.blocks[idx].norm2.bias.data = bias_tensor
            
            filename = f"D:/weights/audio_spectrogram_transformer.encoder.layer.{idx}.intermediate.dense.weight.pth"
            weight_tensor = torch.load(filename)
            self.model.blocks[idx].mlp.fc1.weight.data = weight_tensor
            filename = f"D:/biases/audio_spectrogram_transformer.encoder.layer.{idx}.intermediate.dense.bias.pth"
            bias_tensor = torch.load(filename)
            self.model.blocks[idx].mlp.fc1.bias.data = bias_tensor           
            
            filename = f"D:/weights/audio_spectrogram_transformer.encoder.layer.{idx}.output.dense.weight.pth"
            weight_tensor = torch.load(filename)
            self.model.blocks[idx].mlp.fc2.weight.data = weight_tensor
            filename = f"D:/biases/audio_spectrogram_transformer.encoder.layer.{idx}.output.dense.bias.pth"
            bias_tensor = torch.load(filename)
            self.model.blocks[idx].mlp.fc2.bias.data = bias_tensor
        
        
        filename = "D:/weights/classifier.layernorm.weight.pth"
        weight_tensor = torch.load(filename)
        self.model.norm_cls.weight.data = weight_tensor
        filename = "D:/biases/classifier.layernorm.bias.pth"
        bias_tensor = torch.load(filename)
        self.model.norm_cls.bias.data = bias_tensor
        
        filename = "D:/weights/classifier.dense.weight.pth"
        weight_tensor = torch.load(filename)
        self.model.head.weight.data = weight_tensor
        filename = "D:/biases/classifier.dense.bias.pth"
        bias_tensor = torch.load(filename)
        self.model.head.bias.data = bias_tensor
        
        ######################################################################
        ###   Finishing the assignment of weights and biases to the model  ###
        ######################################################################       
        
        # torch.save(self.model.state_dict(), 'model_with_weights.pth') #we can just use the state dict after the first assignment
        # model_path = "model_with_weights.pth"
        # self.model.load_state_dict(torch.load(model_path))
        
        self.model.head = torch.nn.Linear(self.model.head.in_features, 5)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.initial_lr)
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _prepare_dataloader(self, x, y, shuffle=False):
        dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader
    
    def train(self, epochs=20, lr=None, freeze=True):
        lr = lr if lr is not None else self.initial_lr
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        best_accuracy = 0.0
        best_epoch = 0
        
        for param in self.model.parameters():
            param.requires_grad = not freeze
        for param in self.model.head.parameters():
            param.requires_grad = True      
        
        for epoch in range(epochs):
            self.model.train()  # Set model to training mode
            total_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, targets) in enumerate(self.train_dataloader):
                data = data.to(self.device)
                targets = targets.to(self.device)
                scores = self.model(data)
                loss = self.criterion(scores, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
    
                # Calculate training accuracy
                _, predicted = scores.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
    
                del data
                del targets
            # Calculate training accuracy for the epoch
            epoch_loss = total_loss / len(self.train_dataloader)
            epoch_accuracy = 100.0 * correct / total
            
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_accuracy:.2f}%")
            
            # Validate the model after each epoch
            if self.test_dataloader:
                test_loss, test_accuracy = self.validate()
                if test_accuracy > best_accuracy:
                    best_accuracy = test_accuracy
                    best_epoch = epoch + 1         
                    
        
        if best_accuracy is not None:
            self.outputs_test=best_accuracy
            print(f"Best model is at epoch {best_epoch} with testing accuracy: {best_accuracy:.2f}%")


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
                del data
                del targets
        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = 100*total_correct / len(self.test_dataloader.dataset)
        print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

def ast_feature_extract(x):
    feature_extractor = ASTFeatureExtractor()
    ft = feature_extractor(x, sampling_rate=16000, padding='max_length',
                           return_tensors='pt')
    return ft['input_values']

#if __name__ == "__main__":








# %%
############################################### audio
model = ViT_Encoder(classifier = True, img_size=[1024, 128], in_chans=1, patch_size = (16, 16), stride = 10, embed_pos = True)


import pickle


test_acc_all = list()
for idx in range(2, 3):
    test_acc = []
    torch.cuda.empty_cache()
    direct=r"C:\Users\user.DESKTOP-HI4HHBR\Downloads\Feature_vision"
    file_name = f"subject_{idx:02d}_aud.pkl"
    file_ = os.path.join(direct, file_name)

    with open(file_, 'rb') as f:
        vis_list2 = pickle.load(f)
        
    tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = vis_list2

    tr_x_aud_ft = ast_feature_extract(tr_x_vis)
    te_x_aud_ft = ast_feature_extract(te_x_vis)
    tr_y_aud=tr_y_vis
    te_y_aud=te_y_vis

    data = [tr_x_aud_ft.unsqueeze(1), tr_y_aud, te_x_aud_ft.unsqueeze(1), te_y_aud]

    Trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=8, num_epochs=10)

    Trainer.train(epochs=10, lr=5e-4, freeze=True)
    Trainer.train(epochs=20, lr=5e-6, freeze=False)

    test_acc.append(Trainer.outputs_test)
    accuracy=Trainer.outputs_test
    f = open("accuracy_aud_transf_vit.txt", "a")
    f.write("\n Subject ")
    f.write(str(idx))
    f.write("\n")
    f.write(f"The accuracy of the {idx}-subject is ")
    f.write(str(accuracy))
    print(f"The accuracy of the {idx}-subject is ")
    print(accuracy)
    f.close()

test_acc_all = np.reshape(np.array(test_acc), (42, 1))


# #%%
# import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:64"



# model.to(torch.device("cuda"))
# ft_x = []
# with torch.no_grad():
#     for i in range(200):
#         print("Memory allocated on CUDA:", torch.cuda.memory_allocated())
#         torch.cuda.empty_cache()
#         print("Memory reserved for caching on CUDA:", torch.cuda.memory_reserved())
        
#         # Move input tensor to GPU and extract features
#         a = te_x_aud_ft[i].unsqueeze(0).unsqueeze(0).to(torch.device("cuda"))
#         b = model.feature(a).to('cpu')
        
#         # Append features to ft_x
#         ft_x.append(b)
        
#         # Clear intermediate tensors from GPU memory
#         del a
#         del b
#         torch.cuda.empty_cache()
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import numpy as np

# features = torch.stack(ft_x).squeeze()  # Remove extra dimensions, if any
# # Extract the first row from each sample
# features_first_row = features[:, 0, :].detach()  # This slices out the first row for all samples and detaches it

# # Convert labels to a NumPy array for plotting
# labels = np.array(te_y_aud)

# # Define the emotion_to_index mapping
# emotion_to_index = {
#     'Neutral': 0,
#     'Happiness': 3,
#     'Sadness': 1,
#     'Anger': 2,
#     'Calmness': 4
# }

# # Reverse mapping to get index to emotion mapping for plotting
# index_to_emotion = {v: k for k, v in emotion_to_index.items()}

# # Use t-SNE to reduce the dimensionality
# tsne = TSNE(n_components=2, random_state=42)
# features_2d = tsne.fit_transform(features_first_row)

# # Plotting
# plt.figure(figsize=(10, 8))
# for i, emotion in index_to_emotion.items():
#     indices = labels == i
#     plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=emotion)
# plt.legend()
# plt.title('t-SNE Visualization of the First Row of Features')
# plt.xlabel('Component 1')
# plt.ylabel('Component 2')
# plt.show()

# # %%
# ####################### # EEG
# from Dataload_eeg import DataLoadEEG


# eeg_loader = DataLoadEEG(subject=1, band=[0.5, 45], fs_orig=500, fs_target=100,
#                               parent_directory=r'D:\EAV')

# data_eeg, data_eeg_y = eeg_loader.data_prepare()

# division_eeg = EAVDataSplit(data_eeg, data_eeg_y)
# del data_eeg
# del data_eeg_y
# [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg] = division_eeg.get_split()
# del division_eeg
# data = [torch.from_numpy(tr_x_eeg).float(), tr_y_eeg, torch.from_numpy(te_x_eeg).float(), te_y_eeg]

# model = ViT_Encoder(classifier = True, img_size=[30, 500], in_chans=1,
#                     patch_size = (60, 1), stride = 1, depth=4, num_heads=4,
#                     embed_eeg = True, embed_pos = False)

# trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=32, num_epochs=10)
# trainer.train()
# del data
# del tr_x_eeg
# del tr_y_eeg


# torch.cuda.empty_cache()
# model.to(torch.device("cuda"))
# ft_x = []
# with torch.no_grad():
#     for i in range(200):
#         print("Memory allocated on CUDA:", torch.cuda.memory_allocated())
#         torch.cuda.empty_cache()
#         print("Memory reserved for caching on CUDA:", torch.cuda.memory_reserved())
        
#         # Move input tensor to GPU and extract features
#         a = torch.from_numpy(te_x_eeg[i]).float().unsqueeze(0).to(torch.device("cuda"))
#         ft_x.append(model.feature(a).cpu())
        
#         # Clear intermediate tensors from GPU memory
#         del a
#         torch.cuda.empty_cache()
        
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import numpy as np

# features = torch.stack(ft_x).squeeze()  # Remove extra dimensions, if any
# # Extract the first row from each sample
# features_first_row = features[:, 0, :].detach()  # This slices out the first row for all samples and detaches it

# # Convert labels to a NumPy array for plotting
# labels = np.array(te_y_eeg)

# # Define the emotion_to_index mapping
# emotion_to_index = {
#     'Neutral': 0,
#     'Happiness': 3,
#     'Sadness': 1,
#     'Anger': 2,
#     'Calmness': 4
# }

# # Reverse mapping to get index to emotion mapping for plotting
# index_to_emotion = {v: k for k, v in emotion_to_index.items()}

# # Use t-SNE to reduce the dimensionality
# tsne = TSNE(n_components=2, random_state=42)
# features_2d = tsne.fit_transform(features_first_row)

# # Plotting
# plt.figure(figsize=(10, 8))
# for i, emotion in index_to_emotion.items():
#     indices = labels == i
#     plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=emotion)
# plt.legend()
# plt.title('t-SNE Visualization of the First Row of Features')
# plt.xlabel('Component 1')
# plt.ylabel('Component 2')
# plt.show()


