import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset, DataLoader

from transformers import ASTFeatureExtractor

import timm

import torch.nn.functional as F
from timm.layers import Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType

from Dataload_audio import DataLoadAudio
from EAV_datasplit import EAVDataSplit
import numpy as np

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
            qkv_bias: bool = False,
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
        q, k = self.q_norm(q), self.k_norm(k)

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
            qkv_bias: bool = False,
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
        self.norm1 = norm_layer(dim)
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

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
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
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, bias=bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x

class ViT_Encoder(nn.Module):
    def __init__(self, img_size=[224, 224], in_chans = 3, patch_size=16, stride = 16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 classifier : bool = False, num_classes = 5, embed_eeg = False, embed_pos = True):
        super().__init__()
        # updated_mh
        #self.num_patches = (img_size // patch_size) ** 2
        self.embed_eeg = embed_eeg
        self.embed_pos = embed_pos
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        num_patches_height = (img_size[0] - patch_size[0]) // stride + 1
        num_patches_width = (img_size[1] - patch_size[1]) // stride + 1
        self.total_patches = num_patches_height * num_patches_width

        self.stride = stride
        self.eeg_embed = EEG_decoder()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, stride = stride)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.total_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)

        self.feature_map = None  # this will contain the ViT feature map (including CLASS token)

        self.blocks = nn.ModuleList([
            timm.models.vision_transformer.Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        if classifier:
            self.head = nn.Linear(embed_dim, num_classes, bias=True)
        else:
            self.head = []

    def feature(self, x): # Returns the transformer feature map for all input data, bypassing the classifier head.
        B = x.shape[0]
        if self.embed_eeg:  # Only for EEG
            x = self.eeg_embed(x)
            x = x.unsqueeze(1)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # Copy
        x = torch.cat((cls_tokens, x), dim=1)  # Add class token

        if self.embed_pos:
            x += self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x) # Return the feature map including the class token
        return x


    def forward(self, x):
        B = x.shape[0]
        if self.embed_eeg: ## only for the EEG
            x = self.eeg_embed(x)
            x = x.unsqueeze(1)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # 복제
        x = torch.cat((cls_tokens, x), dim=1)  # 클래스 토큰 추가

        if self.embed_pos:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        self.feature_map = x

        if self.head:  # classifier mode
            x = self.head(x[:, 0])
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

                if batch_idx % 5 == 0:
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
def ast_feature_extract(x):
    feature_extractor = ASTFeatureExtractor()
    ft = feature_extractor(x, sampling_rate=16000, padding='max_length',
                           return_tensors='pt')
    return ft['input_values']


model = ViT_Encoder(classifier = True, img_size=[1024, 128], in_chans=1, patch_size = (16, 16), stride = 10, embed_pos = True)

aud_loader = DataLoadAudio(subject=1, parent_directory=r'C:\\Users\\minho.lee\\Dropbox\\EAV')
[data_aud , data_aud_y] = aud_loader.process()
division_aud = EAVDataSplit(data_aud, data_aud_y)
[tr_x_aud, tr_y_aud, te_x_aud , te_y_aud] = division_aud.get_split()
tr_x_aud_ft = ast_feature_extract(tr_x_aud)
te_x_aud_ft = ast_feature_extract(te_x_aud)
data = [tr_x_aud_ft.unsqueeze(1), tr_y_aud, te_x_aud_ft.unsqueeze(1), te_y_aud]

trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=8, num_epochs=30)
trainer.train()





from Dataload_eeg import DataLoadEEG


eeg_loader = DataLoadEEG(subject=1, band=[0.5, 45], fs_orig=500, fs_target=100,
                             parent_directory=r'C:\\Users\\minho.lee\\Dropbox\\EAV')
data_eeg, data_eeg_y = eeg_loader.data_prepare()

division_eeg = EAVDataSplit(data_eeg, data_eeg_y)
[tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg] = division_eeg.get_split()
data = [torch.from_numpy(tr_x_eeg).float(), tr_y_eeg, torch.from_numpy(te_x_eeg).float(), te_y_eeg]

model = ViT_Encoder(classifier = True, img_size=[30, 500], in_chans=1,
                    patch_size = (60, 1), stride = 1, depth=2, num_heads=4,
                    embed_eeg = True, embed_pos = False)

trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=32, num_epochs=30)
trainer.train()







if __name__ == "__main__":
    # vision
    data = [torch.randn(10, 3, 224, 224), np.random.randint(0, 5, size=(10,)),
            torch.randn(10, 3, 224, 224), np.random.randint(0, 5, size=(10,))] # batch, c, h, w
    model = ViT_Encoder(classifier=True, img_size=(224, 224), in_chans=3, patch_size=(16, 16), stride=16,
                        embed_pos=True)
    trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=8, num_epochs=50)
    trainer.train()

    # audio
    data = [torch.randn(10, 1, 224, 224), np.random.randint(0, 5, size=(10,)),
            torch.randn(10, 1, 224, 224), np.random.randint(0, 5, size=(10,))] # batch, 1, h, w
    model = ViT_Encoder(classifier=True, img_size=(224, 224), in_chans=1, patch_size=(16, 16), stride=16,
                        embed_pos=True)
    trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=8, num_epochs=50)
    trainer.train()

    # eeg : applied 1d cnn, input is 2 ch and Time
    data = [torch.randn(10, 30, 500), np.random.randint(0, 5, size=(10,)),
            torch.randn(10, 30, 500), np.random.randint(0, 5, size=(10,))] # batch, c, h, w
    model = ViT_Encoder(classifier=True, img_size=(30, 500), in_chans=1, patch_size=(16, 16), stride=16,
                        embed_pos=True)
    trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=8, num_epochs=50)
    trainer.train()


