import os
import torchaudio
from torchaudio.transforms import Resample
from transformers import ASTFeatureExtractor
import numpy as np

from EAV_datasplit import *
import Transformer_Audio

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
    for sub_idx in range(1,43):
        aud_loader = DataLoadAudio(subject=sub_idx, parent_directory=r'C:\Users\minho.lee\Dropbox\Datasets\EAV')
        [data_aud , data_aud_y] = aud_loader.process()
        # audio_loader.label_emotion()

        division_aud = EAVDataSplit(data_aud, data_aud_y)
        [tr_x_aud, tr_y_aud, te_x_aud , te_y_aud] = division_aud.get_split()
        data = [tr_x_aud, tr_y_aud, te_x_aud , te_y_aud]

        mod_path = os.path.join(os.getcwd(), 'ast-finetuned-audioset')
        Trainer = Transformer_Audio.AudioModelTrainer(data,  model_path=mod_path, sub = f"subject_{sub_idx:02d}",
                                            num_classes=5, weight_decay=1e-5, lr=0.005, batch_size = 8)

        Trainer.train(epochs=10, lr=5e-4, freeze=True)
        Trainer.train(epochs=15, lr=5e-6, freeze=False)

        test_acc.append(Trainer.outputs_test)




''' test it with the current data
    import pickle
    with open("test_acc_audio.pkl", 'wb') as f:
        pickle.dump(test_acc, f)




with open("test_acc_audio.pkl", 'rb') as f:
    testacc = pickle.load(f)

    # test accuracy for 200 trials
    ## acquire the test label from one subject, it is same for all subjects
    from sklearn.metrics import f1_score
    file_name = f"subject_{1:02d}_vis.pkl"
    file_ = os.path.join(os.getcwd(), 'Feature_vision', file_name)
    with open(file_, 'rb') as f:
        vis_list2 = pickle.load(f)
    te_y_vis = vis_list2[3]

    # load test accuracy for all subjects: 5000 (200, 25) predictions
    with open("test_acc_vision.pkl", 'rb') as f:
        testacc = pickle.load(f)

    test_acc_all = list()
    test_f1_all = list()
    for sub in range(42):
        aa = testacc[sub]
        out1 = np.argmax(aa, axis = 1)
        accuracy = np.mean(out1 == te_y_vis)
        test_acc_all.append(accuracy)

        f1 = f1_score(te_y_vis, out1, average='weighted')
        test_f1_all.append(f1)

    test_acc_all = np.reshape(np.array(test_acc_all), (42, 1))
    test_f1_all = np.reshape(np.array(test_f1_all), (42, 1))





model = AutoModelForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

test_data = torch.tensor(data, dtype=torch.float32)
test_data = test_data.to(device)
aa = test_data[0:20]
with torch.no_grad(): # 572 classes. 
    logits = model(aa).logits

probs = torch.nn.functional.softmax(logits, dim=-1)
predicted_class_id = probs.argmax(dim=1)
bb = np.array(probs.cpu())
config = model.config
config.num_labels
'''