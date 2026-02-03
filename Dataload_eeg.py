"""
EEG Data Processing and Classification Pipeline
===============================================
This script handles loading, preprocessing (downsampling, bandpass filtering),
and splitting of EEG data. It also includes examples of how to train
Transformer and CNN models using PyTorch and TensorFlow.

Dependencies:
    - scipy
    - numpy
    - sklearn
    - torch (optional)
    - tensorflow (optional)
    - EAV_datasplit (local module)
"""
import os
import pickle
import numpy as np
import scipy.io
from scipy import signal
from scipy.signal import butter
from sklearn.metrics import accuracy_score, confusion_matrix

# Local imports
from EAV_datasplit import EAVDataSplit

# --- Constants ---
# NEU_SPE = 108, 0
# S_SPE = 1
# A_SPE = 2
# H_SPE = 3
# R_SPE = 4
SELECTED_CLASSES = [1, 3, 5, 7, 9]  # Classes corresponding to listening tasks

class DataLoadEEG:
    """
    A class to load and preprocess EEG data for a specific subject.
    """
    def __init__(self, subject=1, band=[0.3, 50], fs_orig=500, fs_target=100,
                 parent_directory='./Datasets/EAV'):
        self.subject = subject
        self.band = band
        self.fs_orig = fs_orig
        self.fs_target = fs_target
        self.parent_directory = parent_directory

        # Placeholders for data
        self.seg = None  # Raw segmented data
        self.label = None  # Raw labels
        self.seg_f = None  # Filtered data
        self.seg_f_div = None  # Divided/processed data
        self.label_div = None  # Processed labels

    def load_mat_data(self):
        """Loads .mat files for EEG signal and labels."""
        subject_str = f'subject{self.subject:02d}'
        eeg_folder = os.path.join(self.parent_directory, subject_str, 'EEG')

        # Construct file paths
        # Handle potential naming inconsistencies (double underscores)
        base_name = subject_str.rstrip('__')
        eeg_file_path = os.path.join(eeg_folder, base_name + '_eeg.mat')
        label_file_path = os.path.join(eeg_folder, base_name + '_eeg_label.mat')

        if not os.path.exists(eeg_file_path):
            print(f'[Error] EEG data not found for {subject_str}')
            return

        # Load EEG signal
        mat = scipy.io.loadmat(eeg_file_path)
        if 'seg1' in mat:
            cnt_ = np.array(mat.get('seg1'))
        else:
            cnt_ = np.array(mat.get('seg'))

        # Load Labels
        mat_y = scipy.io.loadmat(label_file_path)
        self.label = np.array(mat_y.get('label'))

        # Transpose to (Channels, Time, Trials)
        # Original: (10000, 30, 200) -> Target: (30, 10000, 200)
        self.seg = np.transpose(cnt_, [1, 0, 2])
        print(f'[Info] Loaded EEG data for {subject_str}')

    def downsampling(self):
        """Downsamples the EEG data from fs_orig to fs_target."""
        if self.seg is None:
            return

        ch, t, tri = self.seg.shape
        factor = self.fs_target / self.fs_orig

        # Reshape for efficient processing: (Channels, Time * Trials)
        tm = np.reshape(self.seg, [ch, t * tri], order='F')

        # Resample
        down_factor = int(self.fs_orig / self.fs_target)
        tm2 = signal.resample_poly(tm, up=1, down=down_factor, axis=1)

        # Reshape back: (Channels, New_Time, Trials)
        new_time = int(t * factor)
        self.seg = np.reshape(tm2, [ch, new_time, tri], order='F')

    def bandpass_filter(self):
        """Applies a Butterworth bandpass filter."""
        if self.seg is None:
            return

        ch, t, tri = self.seg.shape
        dat = np.reshape(self.seg, [ch, t * tri], order='F')

        # Create SOS filter
        sos = butter(5, self.band, btype='bandpass', fs=self.fs_target, output='sos')

        # Apply filter channel by channel
        fdat = []
        for i in range(ch):
            tm = signal.sosfilt(sos, dat[i, :])
            fdat.append(tm)

        self.seg_f = np.array(fdat).reshape((ch, t, tri), order='F')

    def segment_and_select_classes(self):
        """
        Divides the continuous data into smaller windows and selects specific classes.
        Assumes original trial length is 20s (2000 samples @ 100Hz), split into 4 chunks.
        """
        if self.seg_f is None:
            return

        # Reshape: Split 20s trials into 4 x 5s chunks
        # (30, 2000, 200) -> (30, 500, 4, 200)
        tm1 = self.seg_f.reshape((30, 500, 4, 200), order='F')

        # Flatten chunks: (30, 500, 800)
        self.seg_f_div = tm1.reshape((30, 500, 4 * 200), order='F')

        # Repeat labels for the new chunks
        self.label_div = np.repeat(self.label, repeats=4, axis=1)

        # Filter for selected listening classes only
        selected_mask = np.isin(np.argmax(self.label_div, axis=0), SELECTED_CLASSES)

        label_subset = self.label_div[:, selected_mask]
        data_subset = self.seg_f_div[:, :, selected_mask]

        # Transpose data for model input: (Trials, Channels, Time)
        # (30, 500, 400) -> (400, 30, 500)
        self.seg_f_div = np.transpose(data_subset, (2, 0, 1))

        # Convert one-hot labels to class indices
        self.label_div = np.argmax(label_subset, axis=0)

    def prepare_data(self):
        """Pipeline to execute all preprocessing steps."""
        self.load_mat_data()
        self.downsampling()
        self.bandpass_filter()
        self.segment_and_select_classes()
        return self.seg_f_div, self.label_div

# --- Main Execution Block ---

if __name__ == "__main__":
    # Settings
    SUBJECT_RANGE = range(1, 43)
    DATA_ROOT = r'C:\Users\minho.lee\Dropbox\Datasets\EAV'
    OUTPUT_DIR = os.path.join(DATA_ROOT, 'Input_images', 'EEG')

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for sub in SUBJECT_RANGE:
        print(f"\nProcessing Subject {sub}...")

        # 1. Load and Preprocess Data
        eeg_loader = DataLoadEEG(subject=sub, band=[0.5, 45],
                                 fs_orig=500, fs_target=100,
                                 parent_directory=DATA_ROOT)

        data_eeg, data_eeg_y = eeg_loader.prepare_data()

        if data_eeg is None:
            continue

        # 2. Split Data (Train/Test)
        division_eeg = EAVDataSplit(data_eeg, data_eeg_y)
        # Assuming h_idx determines the split ratio or holdout
        tr_x, tr_y, te_x, te_y = division_eeg.get_split(h_idx=56)

        dataset = [tr_x, tr_y, te_x, te_y]

        # Optional: Save processed data to pickle
        # pickle_path = os.path.join(OUTPUT_DIR, f"subject_{sub:02d}_eeg.pkl")
        # with open(pickle_path, 'wb') as f:
        #     pickle.dump(dataset, f)

        # ---------------------------------------------------------
        # MODEL 1: PyTorch Transformer (Custom Implementation)
        # ---------------------------------------------------------
        # from Transformer_torch import Transformer_EEG
        # print("Training Transformer (PyTorch)...")
        # model_trans = Transformer_EEG.EEGClassificationModel(eeg_channel=30)
        # trainer_trans = Transformer_EEG.EEGModelTrainer(dataset, model=model_trans, lr=0.001, batch_size=64)
        # trainer_trans.train(epochs=100, lr=None, freeze=False)
        # acc_trans, preds_trans = trainer_trans.evaluate()
        # print(f"Transformer Accuracy: {acc_trans}")

        # ---------------------------------------------------------
        # MODEL 2: TensorFlow CNN (EEGNet)
        # ---------------------------------------------------------
        # from CNN_tensorflow.CNN_EEG_tf import EEGNet
        # import tensorflow as tf

        # print("Training EEGNet (TensorFlow)...")
        # # Prepare One-Hot Encoding
        # num_classes = 5
        # y_train_oh = np.eye(num_classes)[tr_y.flatten()]
        # y_test_oh = np.eye(num_classes)[te_y.flatten()]

        # # Add channel dimension (N, Ch, T, 1)
        # x_train_tf = tr_x[..., np.newaxis]
        # x_test_tf = te_x[..., np.newaxis]

        # model_tf = EEGNet(nb_classes=num_classes, D=8, F2=64, Chans=30,
        #                   kernLength=300, Samples=500, dropoutRate=0.5)

        # model_tf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model_tf.fit(x_train_tf, y_train_oh, batch_size=32, epochs=200,
        #              shuffle=True, validation_data=(x_test_tf, y_test_oh), verbose=0)

        # pred_probs = model_tf.predict(x_test_tf)
        # pred_labels = np.argmax(pred_probs, axis=1)
        # true_labels = np.argmax(y_test_oh, axis=1)

        # acc_tf = accuracy_score(true_labels, pred_labels)
        # print(f"TF EEGNet Accuracy: {acc_tf}")

        # ---------------------------------------------------------
        # MODEL 3: PyTorch CNN (EEGNet)
        # ---------------------------------------------------------
        try:
            import torch
            import torch.nn as nn
            from CNN_torch.EEGNet_tor import EEGNet_tor, Trainer_uni

            print("Training EEGNet (PyTorch)...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model_torch = EEGNet_tor(nb_classes=5, D=8, F2=64, Chans=30,
                                     kernLength=300, Samples=500, dropoutRate=0.5)

            # Custom trainer class usage
            trainer_torch = Trainer_uni(model=model_torch, data=dataset,
                                        lr=1e-5, batch_size=32, num_epochs=200)
            trainer_torch.train()

            # Evaluation
            model_torch.eval()
            te_x_tensor = torch.tensor(te_x, dtype=torch.float32).to(device)
            te_y_tensor = torch.tensor(te_y, dtype=torch.long).to(device)
            model_torch.to(device)

            with torch.no_grad():
                scores = model_torch(te_x_tensor)
                predictions = scores.argmax(dim=1)
                correct = (predictions == te_y_tensor).sum().item()
                acc_torch = correct / te_y_tensor.size(0)

            print(f"PyTorch EEGNet Accuracy: {acc_torch:.4f}")

        except ImportError as e:
            print(f"Skipping PyTorch training: {e}")
        except Exception as e:
            print(f"Error during PyTorch training: {e}")