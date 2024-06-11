import os
import scipy.io
import numpy as np
from scipy.signal import butter
from scipy import signal

from EAV_datasplit import *

import Transformer_EEG
'''
NEU_SPE = 108, 0
S_SPE = 1
A_SPE = 2
H_SPE = 3
R_SPE = 4  #####
'''

class DataLoadEEG:
    def __init__(self, subject='all', band=[0.3, 50], fs_orig=500, fs_target=100,
                 parent_directory=r'C:\Users\minho.lee\Dropbox\Datasets\EAV'):
        self.subject = subject
        self.band = band
        self.parent_directory = parent_directory
        self.fs_orig = fs_orig
        self.fs_target = fs_target
        self.seg = []
        self.label = []
        self.label_div = []
        self.seg_f = []
        self.seg_f_div = []

    def data_mat(self):
        subject = f'subject{self.subject:02d}'
        eeg_folder = os.path.join(self.parent_directory, subject, 'EEG')
        eeg_file_name = subject.rstrip('__') + '_eeg.mat'
        eeg_file_path = os.path.join(eeg_folder, eeg_file_name)

        label_file_name = subject.rstrip('__') + '_eeg_label.mat'
        label_file_path = os.path.join(eeg_folder, label_file_name)

        if os.path.exists(eeg_file_path):
            mat = scipy.io.loadmat(eeg_file_path)
            cnt_ = np.array(mat.get('seg1'))
            if np.ndim(cnt_) == 3:
                cnt_ = np.array(mat.get('seg1'))
            else:
                cnt_ = np.array(mat.get('seg'))

            mat_y = scipy.io.loadmat(label_file_path)
            label = np.array(mat_y.get('label'))

            self.seg = np.transpose(cnt_, [1, 0, 2])  # (10000, 30, 200) -> (30ch, 10000t, 200trial)
            self.label = label

            print(f'Loaded EEG data for {subject}')
        else:
            print(f'EEG data not found for {subject}')

    def downsampling(self, fs_target=100):
        [ch, t, tri] = self.seg.shape
        factor = fs_target / self.fs_orig
        tm = np.reshape(self.seg, [ch, t * tri], order='F')
        tm2 = signal.resample_poly(tm, up=1, down=int(self.fs_orig / fs_target), axis=1)
        self.seg = np.reshape(tm2, [ch, int(t * factor), tri], order='F')

    def bandpass(self):
        [ch, t, tri] = self.seg.shape
        dat = np.reshape(self.seg, [ch, t * tri], order='F')
        # bandpass after the downsample  -> fs_target
        sos = butter(5, self.band, btype='bandpass', fs=self.fs_target, output='sos')
        fdat = list()
        for i in range(np.size(dat, 0)):
            tm = signal.sosfilt(sos, dat[i, :])
            fdat.append(tm)
        self.seg_f = np.array(fdat).reshape((ch, t, tri), order='F')

    def data_div(self):
        # Here 2000 (20seconds) are divided into 4 splits
        [ch, t, tri] = self.seg_f.shape
        tm1 = self.seg_f.reshape((30, 500, 4, 200), order='F')
        self.seg_f_div = tm1.reshape((30, 500, 4 * 200), order='F')
        self.label_div = np.repeat(self.label, repeats=4, axis=1)

        # Here we only select the listening classes
        selected_classes = [1, 3, 5, 7, 9]
        label = self.label_div[selected_classes, :]
        selected_indices = np.isin(np.argmax(self.label_div, axis=0), selected_classes)
        label = label[:, selected_indices]
        x = self.seg_f_div[:, :, selected_indices]


        self.seg_f_div = np.transpose(x, (2, 0, 1))  # (30, 500, 400) -> (400, 30, 500)
        class_indices = np.argmax(label, axis=0)

        #self.label_div = label
        self.label_div = class_indices

    def data_split(self):
        selected_classes = [1, 3, 5, 7, 9]  # only listening classes
        label = self.label_div[selected_classes, :]

        selected_indices = np.isin(np.argmax(self.label_div, axis=0), selected_classes)
        label = label[:, selected_indices]

        x = self.seg_f_div[:, :, selected_indices]
        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []

        for i in range(5):  # Looping over each class
            class_indices = np.where(label.T[:, i] == 1)[0]  # Find indices where current class label is 1
            midpoint = len(class_indices) // 2  # Calculate the midpoint for 50% split

            # Split data based on found indices
            x_train_list.append(x[:, :, class_indices[:midpoint]])
            x_test_list.append(x[:, :, class_indices[midpoint:]])

            y_train_list.append(label.T[class_indices[:midpoint]])
            y_test_list.append(label.T[class_indices[midpoint:]])

        # Convert lists to numpy arrays
        x_train = np.concatenate(x_train_list, axis=0)
        x_test = np.concatenate(x_test_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

    def data_prepare(self):
        self.data_mat()
        self.downsampling()
        self.bandpass()
        self.data_div()
        return self.seg_f_div, self.label_div


''' Direct evaluation
if __name__ == "__main__":
    eeg_loader = DataLoadEEG(subject=1, band=[0.5, 45], fs_orig=500, fs_target=100,
                             parent_directory='C://Users//minho.lee//Dropbox//Datasets//EAV')
    data_eeg, data_eeg_y = eeg_loader.data_prepare()

    division_eeg = EAVDataSplit(data_eeg, data_eeg_y)
    [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg] = division_eeg.get_split()
    data = [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg]

    trainer = Transformer_EEG.EEGModelTrainer(data, lr=0.001, batch_size = 64)
    trainer.train(epochs=200, lr=None, freeze=False)
'''
from Transformer_EEG import EEGClassificationModel
accuracy_all = list()
prediction_all = list()
if __name__ == "__main__": # from pickle data
    import pickle
    for sub in range(1, 43):
        file_path = "C:/Users/minho.lee/Dropbox/Datasets/EAV/Input_images/EEG/"
        file_name = f"subject_{sub:02d}_eeg.pkl"
        file_ = os.path.join(file_path, file_name)

        with open(file_, 'rb') as f:
            eeg_list2 = pickle.load(f)
        tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg = eeg_list2
        data = [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg]

        model = EEGClassificationModel(eeg_channel=30)
        trainer = Transformer_EEG.EEGModelTrainer(data, model = model, lr=0.001, batch_size = 64)
        trainer.train(epochs=100, lr=None, freeze=False)

        [accuracy, predictions] = trainer.evaluate()
        accuracy_all.append(accuracy)
        prediction_all.append(predictions)

''' create eeg pickle files
if __name__ == "__main__":
    for sub in range(1,43):
        print(sub)
        file_path = "C:/Users/minho.lee/Dropbox/Datasets/EAV/Input_images/EEG/"
        file_name = f"subject_{sub:02d}_eeg.pkl"
        file_ = os.path.join(file_path, file_name)
        eeg_loader = DataLoadEEG(subject=sub, band=[0.5, 45], fs_orig=500, fs_target=100,
                                 parent_directory='C://Users//minho.lee//Dropbox//Datasets//EAV')
        data_eeg, data_eeg_y = eeg_loader.data_prepare()

        division_eeg = EAVDataSplit(data_eeg, data_eeg_y)
        [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg] = division_eeg.get_split()
        EEG_list = [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg]
        import pickle
        with open(file_, 'wb') as f:
            pickle.dump(EEG_list, f)
'''