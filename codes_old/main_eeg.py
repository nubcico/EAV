from Transformer_torch import Transformer_EEG
import numpy as np
import os

fold = os.path.join(os.getcwd(), 'Feature_eeg', 'eeg_data_all.pkl')
import pickle
with open(fold, 'rb') as f:
    data = pickle.load(f)

trainer = Transformer_EEG.EEGModelTrainer(data, lr=0.001, batch_size=128)
model = trainer.train(epochs=40, lr=None, freeze=False)

[tr_x_all, tr_y_all, te_x_all, te_y_all] = data
tr_x = np.reshape(tr_x_all, (200, 42, 30, 500), 'F')
te_x = np.reshape(te_x_all, (200, 42, 30, 500), 'F')

tr_y = np.reshape(tr_y_all, (200, 42), 'F')
te_y = np.reshape(tr_y_all, (200, 42), 'F')

test_acc = []
for i in range(0, 42):
    print(i)
    tr_xx, tr_yy, te_xx, te_yy = tr_x[:, i, :, :], tr_y[:, i], te_x[:, i, :, :], te_y[:, i]
    trainer = Transformer_EEG.EEGModelTrainer([tr_xx, tr_yy, te_xx, te_yy], model = [], lr=0.001, batch_size=32)
    trainer.train(epochs=50, lr=None, freeze=False)
    test_acc.append(trainer.test_acc)

a = 33
