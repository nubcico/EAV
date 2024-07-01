# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:37:59 2024

@author: madina.kudaibergenova

ShallowNet and DeepConvNet

"""
import os
import pickle
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras import Model as KerasModel
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam


def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


class ShallowConvNet:
    @staticmethod
    def build(nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
        input_main = Input((Chans, Samples, 1))
        block1 = Conv2D(40, (1, 13), input_shape=(Chans, Samples, 1),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
        block1 = Conv2D(40, (Chans, 1), use_bias=False,
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
        block1 = Activation(square)(block1)
        block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
        block1 = Activation(log)(block1)
        block1 = Dropout(dropoutRate)(block1)
        flatten = Flatten()(block1)
        dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation('softmax')(dense)

        return KerasModel(inputs=input_main, outputs=softmax)


class DeepConvNet:
    @staticmethod
    def build(nb_classes, Chans=64, Samples=256, dropoutRate=0.5):
        input_main = Input((Chans, Samples, 1))
        block1 = Conv2D(25, (1, 5), input_shape=(Chans, Samples, 1),
                        kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
        block1 = Conv2D(25, (Chans, 1), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block1 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block1)
        block1 = Activation('elu')(block1)
        block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
        block1 = Dropout(dropoutRate)(block1)

        block2 = Conv2D(50, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
        block2 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block2)
        block2 = Activation('elu')(block2)
        block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
        block2 = Dropout(dropoutRate)(block2)

        block3 = Conv2D(100, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
        block3 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block3)
        block3 = Activation('elu')(block3)
        block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
        block3 = Dropout(dropoutRate)(block3)

        block4 = Conv2D(200, (1, 5), kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
        block4 = BatchNormalization(epsilon=1e-05, momentum=0.1)(block4)
        block4 = Activation('elu')(block4)
        block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
        block4 = Dropout(dropoutRate)(block4)

        flatten = Flatten()(block4)
        dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
        softmax = Activation('softmax')(dense)

        return KerasModel(inputs=input_main, outputs=softmax)


def load_eeg_data(file_path):
    eeg_list_all = []
    for i in range(1, 43):
        file_name = f"subject_{i:02d}_eeg.pkl"
        file_ = os.path.join(file_path, file_name)
        print(i)

        if not os.path.isfile(file_):
            print(f"File {file_} does not exist.")
            continue
        if os.path.getsize(file_) == 0:
            print(f"File {file_} is empty.")
            continue

        try:
            with open(file_, 'rb') as f:
                eeg_list = pickle.load(f)
        except EOFError as e:
            print(f"Error loading {file_}: {e}")
            continue

        eeg_list_all.append(eeg_list)
    return eeg_list_all


def train_and_evaluate(models, eeg_list_all, checkpoint_dir, epochs=350, batch_size=32, desired_subject_indices=[0]):
    for model_name, model_class in models.items():
        model_saved = []
        model_loss_list = []
        model_acc_list = []

        for i, (x_tr, y_tr, x_te, y_te) in enumerate(eeg_list_all):
            if i not in desired_subject_indices:
                continue
            model = model_class.build(nb_classes=5, Chans=30, Samples=500)
            model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, validation_data=(x_te, y_te), verbose=1)
            model_saved.append(model)
            model.save(os.path.join(checkpoint_dir, f'model_{model_name}_subj{i}.h5'))
            loss, accuracy = model.evaluate(x_te, y_te)
            model_loss_list.append(loss)
            model_acc_list.append(accuracy)
            print(f'{model_name} Subject {i} - Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')


if __name__ == "__main__":
    
    FILE_PATH = ".../Dropbox/DATASETS/EAV/Input_images/EEG/"
    CHECKPOINT_DIR = '...'
    
    eeg_list_all = load_eeg_data(FILE_PATH)

    models = {
        "shallow": ShallowConvNet,
        "deep": DeepConvNet
    }

    train_and_evaluate(models, eeg_list_all, CHECKPOINT_DIR)
