from Dataload_audio import DataLoadAudio
from EAV_datasplit import EAVDataSplit
from Fusion.VIT_audio.Transformer_audio import ViT_Encoder
from Fusion.VIT_audio.Transformer_audio import ast_feature_extract
from Fusion.VIT_audio.Transformer_audio import Trainer_uni
import torch
import numpy as np

sub_idx = [2, 4, 5, 9, 15, 17, 18, 20, 33, 39]

train_tensors = []
train_labels = []
test_tensors = []
test_labels = []

for sub in sub_idx:
    aud_loader = DataLoadAudio(subject=sub, parent_directory='C:\\Users\\minho.lee\\Dropbox\\Datasets\\EAV')
    [data_aud, data_aud_y] = aud_loader.process()
    division_aud = EAVDataSplit(data_aud, data_aud_y)
    [tr_x_aud, tr_y_aud, te_x_aud, te_y_aud] = division_aud.get_split()
    tr_x_aud_ft = ast_feature_extract(tr_x_aud)
    te_x_aud_ft = ast_feature_extract(te_x_aud)

    # Unsqueezing and storing the tensors in the list
    train_tensors.append(tr_x_aud_ft.unsqueeze(1))
    train_labels.append(tr_y_aud)
    test_tensors.append(te_x_aud_ft.unsqueeze(1))
    test_labels.append(te_y_aud)


data = [torch.cat(train_tensors, dim=0), np.concatenate(train_labels, axis=0),
        torch.cat(test_tensors, dim=0), np.concatenate(test_labels, axis=0)]

model = ViT_Encoder(classifier=True, img_size=[1024, 128], in_chans=1, patch_size=(16, 16), stride=10,
                    embed_pos=True)
trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=8, num_epochs=30)
trainer.train()