import os
import cv2
import numpy as np
import EAV_datasplit
from facenet_pytorch import MTCNN
import torch


class DataLoadVision:
    def __init__(self, subject='all', parent_directory=r'C:\Users\minho.lee\Dropbox\EAV', face_detection=False,
                 image_size=224):
        self.IMG_HEIGHT, self.IMG_WIDTH = 480, 640
        self.subject = subject
        self.parent_directory = parent_directory
        self.file_path = list()
        self.file_emotion = list()
        self.images = list()
        self.image_label = list()  # actual class name
        self.image_label_idx = list()
        self.face_detection = face_detection
        self.image_size = image_size
        self.face_image_size = 56 #
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.mtcnn = MTCNN(
            image_size=self.face_image_size, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            device=self.device
        )

    def data_files(self):
        subject = f'subject{self.subject:02d}'
        print(subject, " Loading")
        file_emotion = []
        subjects = []
        path = os.path.join(self.parent_directory, subject, 'Video')
        for i in os.listdir(path):
            emotion = i.split('_')[4]
            self.file_emotion.append(emotion)
            self.file_path.append(os.path.join(path, i))

    def data_load(self):

        for idx, file in enumerate(self.file_path):
            nm_class = file.split("_")[-1].split(".")[0]  # we extract the class label from the file

            if "Speaking" in file and file.endswith(".mp4"):
                print(idx)
                cap = cv2.VideoCapture(file)
                # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # ~600
                # frame_rate = cap.get(cv2.CAP_PROP_FPS) # 30 frame
                a1 = []
                if cap.isOpened():
                    frame_index = 1
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # (30 framerate * 20s) * 100 Speaking, Select every 6th frame from the first 600 frames
                        # face detection, we converted it into 0-255 again from the [-1, 1] tensor, you can directly return the tensor
                        if (frame_index - 1) % 6 == 0 and frame_index <= 600:
                            if self.face_detection:
                                with torch.no_grad():
                                    x_aligned, prob = self.mtcnn(frame, return_prob=True)
                                    if prob > 0.3:
                                        x_aligned = (x_aligned + 1) / 2
                                        x_aligned = np.clip(x_aligned * 255, 0, 255)
                                        x_aligned = np.transpose(x_aligned.numpy().astype('uint8'), (1, 2, 0))
                                        a1.append(x_aligned)
                                    else:
                                        print("Face is not detected, original is saved")
                                        a1.append(x_aligned)  # incase that face has not been detected, add previous one
                                    pass
                            else:
                                resizedImg = cv2.resize(frame, (self.image_size, self.image_size))
                                a1.append(resizedImg) # sabina: dlkfjefoie

                            if len(a1) == 25:  # 25 frame is 5s each
                                self.images.append(a1)  # this will contain 400 samples [400, 25, (225, 225, 3)]
                                a1 = []
                                self.image_label.append(nm_class)
                        frame_index += 1
                    cap.release()
                else:
                    print(f"Error opening video file: {file}")
        emotion_to_index = {
            'Neutral': 0,
            'Happiness': 3,
            'Sadness': 1,
            'Anger': 2,
            'Calmness': 4
        }
        self.image_label_idx = [emotion_to_index[emotion] for emotion in self.image_label]

    def process(self):
        self.data_files()
        self.data_load()
        return self.images, self.image_label_idx


if __name__ == '__main__':

    for sub in range(1, 20):
        print(sub)
        file_path = "C:/Users/minho.lee/Dropbox/Datasets/EAV/Input_images/Vision/"
        file_name = f"subject_{sub:02d}_vis.pkl"
        file_ = os.path.join(file_path, file_name)
        #if not os.path.exists(file_):

        vis_loader = DataLoadVision(subject=sub, parent_directory=r'C:\Users\minho.lee\Dropbox\Datasets\EAV', face_detection=True)
        [data_vis, data_vis_y] = vis_loader.process()

        eav_loader = EAV_datasplit.EAVDataSplit(data_vis, data_vis_y)

        #each class contains 80 trials, 5/5 radio (h_idx=40), 7/3 ratio (h_dix=56)
        [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis] = eav_loader.get_split(h_idx=56)  # output(list): train, trlabel, test, telabel
        data = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]

        ''' 
        # Here you can write / load vision features tr:{280}(25, 56, 56, 3), te:{120}(25, 56, 56, 3): trials, frames, height, weight, channel
        import pickle        
        Vis_list = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]
        with open(file_, 'wb') as f:
            pickle.dump(Vis_list, f)
        
        # You can directly work from here        
        with open(file_, 'rb') as f:
            Vis_list = pickle.load(f)
        tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = Vis_list
        data = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]
        '''
        # Transformer for Vision
        from Transformer_torch import Transformer_Vision

        mod_path = os.path.join('C:\\Users\\minho.lee\\Dropbox\\Projects\\EAV', 'facial_emotions_image_detection')
        trainer = Transformer_Vision.ImageClassifierTrainer(data,
                                                            model_path=mod_path, sub=f"subject_{sub:02d}",
                                                            num_labels=5, lr=5e-5, batch_size=128)
        trainer.train(epochs=10, lr=5e-4, freeze=True)
        trainer.train(epochs=5, lr=5e-6, freeze=False)
        trainer.outputs_test

        # CNN for Vision
        from CNN_torch.CNN_Vision import ImageClassifierTrainer
        trainer = ImageClassifierTrainer(data, num_labels=5, lr=5e-5, batch_size=32)
        trainer.train(epochs=3, lr=5e-4, freeze=True)
        trainer.train(epochs=3, lr=5e-6, freeze=False)
        trainer._delete_dataloader()
        trainer.outputs_test






