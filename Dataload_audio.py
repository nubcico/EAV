import os
import torchaudio
from torchaudio.transforms import Resample
from transformers import ASTFeatureExtractor

from EAV_datasplit import *
from Transformer_torch import Transformer_Audio


class DataLoadAudio:
    def __init__(self, subject='all', parent_directory=r'C:\Users\minho.lee\Dropbox\Datasets\EAV', target_sampling_rate=16000):
        self.parent_directory = parent_directory
        self.original_sampling_rate = int()
        self.target_sampling_rate = target_sampling_rate
        self.subject = subject
        self.file_path = list()
        self.file_emotion = list()

        self.seg_length = 5  # 5s
        self.feature = None
        self.label = None
        self.label_indexes = None
        self.test_prediction = list()

    def data_files(self):
        subject = f'subject{self.subject:02d}'
        file_emotion = []
        subjects = []
        path = os.path.join(self.parent_directory, subject, 'Audio')
        for i in os.listdir(path):
            emotion = i.split('_')[4]
            self.file_emotion.append(emotion)
            self.file_path.append(os.path.join(path, i))

    def feature_extraction(self):
        x = []
        y = []
        feature_extractor = ASTFeatureExtractor()
        for idx, path in enumerate(self.file_path):
            waveform, sampling_rate = torchaudio.load(path)
            self.original_sampling_rate = sampling_rate
            if self.original_sampling_rate is not self.target_sampling_rate:
                resampler = Resample(orig_freq=sampling_rate, new_freq=self.target_sampling_rate)
                resampled_waveform = resampler(waveform)
                resampled_waveform = resampled_waveform.squeeze().numpy()
            else:
                resampled_waveform = waveform

            segment_length = self.target_sampling_rate * self.seg_length
            num_sections = int(np.floor(len(resampled_waveform) / segment_length))

            for i in range(num_sections):
                t = resampled_waveform[i * segment_length: (i + 1) * segment_length]
                x.append(t)
                y.append(self.file_emotion[idx])
        print(f"Original sf: {self.original_sampling_rate}, resampled into {self.target_sampling_rate}")

        emotion_to_index = {
            'Neutral': 0,
            'Happiness': 3,
            'Sadness': 1,
            'Anger': 2,
            'Calmness': 4
        }
        y_idx = [emotion_to_index[emotion] for emotion in y]
        self.feature = np.squeeze(np.array(x))
        self.label_indexes = np.array(y_idx)
        self.label = np.array(y)

    def process(self):
        self.data_files()
        self.feature_extraction()
        return self.feature, self.label_indexes

    def label_emotion(self):
        self.data_files()
        self.feature_extraction()
        return self.label

if __name__ == "__main__":
    test_acc = []
    for sub in range(1,43):
        file_path = "C:/Users/minho.lee/Dropbox/Datasets/EAV/Input_images/Audio/"
        file_name = f"subject_{sub:02d}_aud.pkl"
        file_ = os.path.join(file_path, file_name)

        aud_loader = DataLoadAudio(subject=sub, parent_directory=r'C:\Users\minho.lee\Dropbox\Datasets\EAV')
        [data_aud , data_aud_y] = aud_loader.process()
        # audio_loader.label_emotion()

        division_aud = EAVDataSplit(data_aud, data_aud_y)
        [tr_x_aud, tr_y_aud, te_x_aud , te_y_aud] = division_aud.get_split(h_idx=56)
        data = [tr_x_aud, tr_y_aud, te_x_aud , te_y_aud]

        ''' 
        # Here you can write / load vision features tr:{280}(80000), te:{120}(80000): trials, frames, height, weight, channel
        # This code is to store the RAW audio input to the folder: (400, 80000), 16000Hz
        import pickle        
        Aud_list = [tr_x_aud, tr_y_aud, te_x_aud, te_y_aud]
        with open(file_, 'wb') as f:
            pickle.dump(Aud_list, f)

        # You can directly work from here
        with open(file_, 'rb') as f:
            Aud_list = pickle.load(f)
        [tr_x_aud, tr_y_aud, te_x_aud, te_y_aud] = Aud_list
        data = [tr_x_aud, tr_y_aud, te_x_aud , te_y_aud]
        '''
        mod_path = os.path.join(os.getcwd(), 'ast-finetuned-audioset')
        Trainer = Transformer_Audio.AudioModelTrainer(data, model_path=mod_path, sub =f"subject_{sub:02d}",
                                                      num_classes=5, weight_decay=1e-5, lr=0.005, batch_size = 8)

        Trainer.train(epochs=10, lr=5e-4, freeze=True)
        Trainer.train(epochs=15, lr=5e-6, freeze=False)
        test_acc.append(Trainer.outputs_test)

        ## Add CNN - audio here, refer to the file Dataload_vision.py



