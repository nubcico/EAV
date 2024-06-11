import os
import scipy.io
import numpy as np
#from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Activation, Permute, Dropout, Conv2D,
                                    MaxPooling2D, AveragePooling2D, SeparableConv2D,
                                    DepthwiseConv2D, BatchNormalization, SpatialDropout2D,
                                    Input, Flatten)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from scipy import signal
from scipy.signal import butter, sosfilt, sosfreqz
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from scipy.linalg import eigh
import tensorflow as tf
from scipy.stats import entropy
from sklearn.model_selection import train_test_split

# Path to the root directory
parent_directory = r'C:\Users\minho.lee\Dropbox\Datasets\EAV'
# Get all directories in the parent directory that start with "subject"
subject_folders = [d for d in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, d)) and d.startswith("subject")]

# Sort the list
sorted_subject_folders = sorted(subject_folders, key=lambda s: int(s.replace("subject", "")))

print(sorted_subject_folders)
def EEGNet(nb_classes, Chans=64, Samples=128,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength), padding='same',
                    input_shape=(Chans, Samples, 1),
                    use_bias=False)(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D((Chans, 1), use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16),
                             use_bias=False, padding='same')(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax)

    # input = (10000, 30, 200)
    # out = (10000, 30, 200)
def Bandpass(dat, freq=[5, 80], fs=500):
    [D, Ch, Tri] = dat.shape
    dat2 = np.transpose(dat, [0, 2, 1])
    dat3 = np.reshape(dat2, [10000 * 200, 30], order='F')

    sos = butter(5, freq, 'band', fs=fs, output='sos')
    fdat = list()
    for i in range(np.size(dat3, 1)):
        tm = signal.sosfilt(sos, dat3[:, i])
        fdat.append(tm)
    dat4 = np.array(fdat).transpose().reshape((D, Tri, Ch), order='F').transpose(0, 2, 1)
    return dat4
# no need this function anymore
def seg(dat, marker, ival=[0, 2000]):
    sdat = list()
    for i in range(np.size(marker, 1)):
        lag = range(marker[0, i] + ival[0], marker[0, i] + ival[1])
        sdat.append(dat[lag, :])
    return np.array(sdat)
# data = (200, 2000, 30), label = (10, 160)
def mysplit(data, label):
    # Original data and labels
    # data = np.random.rand(200, 30, 2000)  # Replace with your actual data
    # labels = np.random.randint(0, 2, size=(160, 10))  # Replace with your actual labels (one-hot vectors)

    # Splitting parameters
    split_length = 500  # Length of each split
    num_splits = data.shape[1] // split_length  # Number of splits

    a1 = np.transpose(data, [1, 0, 2])
    a2 = np.reshape(a1, [500, 4, 200, 30], order='F')
    a3 = np.reshape(a2, [500, 4 * 200, 30], order='F')
    a4 = np.transpose(a3, [1, 0, 2])

    labels_repeated = np.repeat(label, repeats=4, axis=1)

    return a4, labels_repeated

result_conf = list()
result_acc = list()
result_f1 = list()

for subject in sorted_subject_folders:
    # Remove trailing '__' if present
    # subject = subject.rstrip('__')
    cnt_ = [];
    # Construct the full path to the EEG file
    eeg_folder = os.path.join(parent_directory, subject, 'EEG')
    eeg_file_name = subject.rstrip('__') + '_eeg.mat'
    eeg_file_path = os.path.join(eeg_folder, eeg_file_name)

    label_file_name = subject.rstrip('__') + '_eeg_label.mat'
    label_file_path = os.path.join(eeg_folder, label_file_name)

    # Check if the EEG file exists
    if os.path.exists(eeg_file_path):
        mat = scipy.io.loadmat(eeg_file_path)
        cnt_ = np.array(mat.get('seg1'))
        if np.ndim(cnt_) == 3:
            cnt_ = np.array(mat.get('seg1'))
        else:
            cnt_ = np.array(mat.get('seg'))

        mat_Y = scipy.io.loadmat(label_file_path)
        Label = np.array(mat_Y.get('label'))

        print(f'Loaded EEG data for {subject}')
    else:
        print(f'EEG data not found for {subject}')

    cnt_f = Bandpass(cnt_, freq=[3, 50], fs=500)

    fs_original = 500  # Original sampling rate in Hz
    fs_target = 100  # Target sampling rate in Hz

    tm = np.transpose(cnt_f, [0, 2, 1]).reshape([10000 * 200, 30], order='F')

    downsampling_factor = fs_target / fs_original
    tm2 = signal.resample_poly(tm, up=1, down=int(fs_original / fs_target), axis=0)
    cnt_f2 = np.reshape(tm2, [2000, 200, 30], order='F')

    cnt_seg = np.transpose(cnt_f2, [1, 0, 2])

    num_trials_per_class = np.sum(Label, axis=1)  # check the balance.
    [cnt_seg_split, Label_split] = mysplit(cnt_seg, Label)

    dat = np.transpose(cnt_seg_split, (0, 2, 1)).reshape(
        (800, 30, 500, 1))  # This should be (800, 30, 500, 1) for balanced classes

    selected_classes = [1, 3, 5, 7, 9]  # only listening classes

    selected_indices = np.isin(np.argmax(Label_split, axis=0), selected_classes)

    data_5class = dat[selected_indices]

    aa = Label_split[:, selected_indices]
    label_5class = aa[selected_classes, :]

    # x_train, x_test, y_train, y_test = train_test_split(data_5class, label_5class.T, test_size=0.5, random_state=42, stratify=label_5class.T)

    # Lists to collect train and test subsets
    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []

    for i in range(5):  # Looping over each class
        class_indices = np.where(label_5class.T[:, i] == 1)[0]  # Find indices where current class label is 1
        midpoint = len(class_indices) // 2  # Calculate the midpoint for 50% split

        # this random shuffle will decide "random order" or "sequential order in time"
        # np.random.shuffle(class_indices)  # Shuffle the indices randomly
        # Split data based on found indices
        x_train_list.append(data_5class[class_indices[:midpoint]])
        x_test_list.append(data_5class[class_indices[midpoint:]])

        y_train_list.append(label_5class.T[class_indices[:midpoint]])
        y_test_list.append(label_5class.T[class_indices[midpoint:]])

    # Convert lists to numpy arrays
    x_train = np.concatenate(x_train_list, axis=0)
    x_test = np.concatenate(x_test_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    model = EEGNet(nb_classes=5, D=8, F2=64, Chans=30, kernLength=300, Samples=500,
                   dropoutRate=0.5)

    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=32, epochs=100, shuffle=True, validation_data=(x_test, y_test))

    pred = model.predict(x_test)
    pred = np.argmax(pred, axis=1)

    y_test2 = np.argmax(y_test, axis=1)

    cm = confusion_matrix(pred, y_test2)

    accuracy = accuracy_score(pred, y_test2)
    f1 = f1_score(y_test2, pred, average='weighted')  # 'weighted' for multiclass F1-Score

    result_conf.append(cm)
    result_acc.append(accuracy)
    result_f1.append(f1)  # Append the F1-Score to your result list
    print(result_acc)

result_conf_np = np.array(result_conf)
summed_confusion_matrix = np.sum(result_conf_np, axis=0)
summed_confusion_matrix_T = summed_confusion_matrix.T