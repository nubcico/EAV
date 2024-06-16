from Fusion.VIT_audio.Transformer_audio import ViT_Encoder, Trainer_uni
import torch
from Dataload_eeg import DataLoadEEG
from EAV_datasplit import EAVDataSplit

import pickle
import os
sub = 1
file_path = "C:/Users/minho.lee/Dropbox/Datasets/EAV/Input_images/EEG/"
file_name = f"subject_{sub:02d}_eeg.pkl"
file_ = os.path.join(file_path, file_name)

with open(file_, 'rb') as f:
    eeg_list2 = pickle.load(f)
tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = eeg_list2
data = [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg]

model_eeg = ViT_Encoder(classifier=True, img_size=[60, 500], in_chans=1,
                        patch_size=(60, 1), stride=1, depth=1, num_heads=1,
                        embed_eeg=True, embed_pos=False)

trainer = Trainer_uni(model=model_eeg, data=data, lr=1e-5, batch_size=16, num_epochs=100)
trainer.train()