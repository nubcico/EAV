from Fusion.VIT_audio.Transformer_audio import ViT_Encoder, Trainer_uni
import torch
from Dataload_eeg import DataLoadEEG
from EAV_datasplit import EAVDataSplit

eeg_loader = DataLoadEEG(subject=2, band=[5, 49], fs_orig=500, fs_target=100,
                             parent_directory=r'C:\Users\minho.lee\Dropbox\Datasets\EAV')
data_eeg, data_eeg_y = eeg_loader.data_prepare()

division_eeg = EAVDataSplit(data_eeg, data_eeg_y)
[tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg] = division_eeg.get_split()
data = [torch.from_numpy(tr_x_eeg).float(), tr_y_eeg, torch.from_numpy(te_x_eeg).float(), te_y_eeg]

model_eeg = ViT_Encoder(classifier=True, img_size=[60, 500], in_chans=1,
                        patch_size=(60, 1), stride=1, depth=1, num_heads=1,
                        embed_eeg=True, embed_pos=False)

trainer = Trainer_uni(model=model_eeg, data=data, lr=1e-5, batch_size=16, num_epochs=100)
trainer.train()