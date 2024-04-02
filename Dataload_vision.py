import os
import cv2
import numpy as np
import EAV_datasplit
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from transformers import AutoImageProcessor

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
                        # face detection, we converted it into 0-255 again from the [-1 - 1] tensor, you can directly return the tensor
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
        file_path = "D:/Dropbox/pythonProject/Feature_vision/"
        file_name = f"subject_{sub:02d}_vis.pkl"
        file_ = os.path.join(file_path, file_name)
        if not os.path.exists(file_):
            vis_loader = DataLoadVision(subject=sub, parent_directory='D:/Dropbox/DATASETS/EAV', face_detection=True)
            [data_vis, data_vis_y] = vis_loader.process()

            eav_loader = EAV_datasplit.EAVDataSplit(data_vis, data_vis_y)
            [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis] = eav_loader.get_split()  # output(list): train, trlabel, test, telabel

            Vis_list = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]
            import pickle
            with open(file_, 'wb') as f:
                pickle.dump(Vis_list, f)



'''
    import Transformer_Vision
    data = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]
    trainer = Transformer_Vision.ImageClassifierTrainer(data,
                                     model_path='C:/Users/minho.lee/Dropbox/zEmotion_fusion/pythonProject/facial_emotions_image_detection',
                                     num_labels=5, lr=5e-5, batch_size=128)

    trainer.train(epochs=3, lr=5e-5, freeze=True)
    trainer.train(epochs=3, lr=5e-6, freeze=False)





Vis_list = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]

import pickle
file_path = "C:/Users/minho.lee/Dropbox/zEmotion_fusion/pythonProject/Feature_vision/"
idx = 2
file_name = f"subject_{idx:02d}_vis.pkl"
file_ = os.path.join(file_path, file_name)

with open(file_, 'wb') as f:
    pickle.dump(Vis_list, f)


with open(file_, 'rb') as f:
    vis_list2 = pickle.load(f)
tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = vis_list2



import Transformer_Vision as vs
trainer = vs.ImageClassifierTrainer([tr_x, tr_y, te_x, te_y], model_path = 'C:/Users/minho.lee/Dropbox/zEmotion_fusion/pythonProject/facial_emotions_image_detection',num_labels=5, lr=5e-5, batch_size=128, freeze=True)
trainer.train(epochs=3)

 # "id2label": {    "0": "sad",    "1": "disgust",    "2": "angry",    "3": "neutral",    "4": "fear",    "5": "surprise",    "6": "happy"   },


import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
processor = AutoImageProcessor.from_pretrained("C:/Users/minho.lee/Dropbox/zEmotion_fusion/pythonProject/facial_emotions_image_detection")

model = AutoModelForImageClassification.from_pretrained("C:/Users/minho.lee/Dropbox/zEmotion_fusion/pythonProject/facial_emotions_image_detection")

# Update the model's classification head
num_labels = 5
model.classifier = torch.nn.Linear(model.config.hidden_size, num_labels)
model.num_labels = num_labels

# Freeze all layers in the model
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the classifier layer
for param in model.classifier.parameters():
    param.requires_grad = True

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Assuming tr_x and te_x are lists or arrays of PIL Images
def preprocess_images(image_list):
    pixel_values_list = []
    for img in image_list:
        processed = processor(images=img, return_tensors="pt")
        pixel_values = processed.pixel_values.squeeze()  # Remove batch dim
        pixel_values_list.append(pixel_values)
    return torch.stack(pixel_values_list)  # Stack into a single tensor

# Preprocess and prepare the datasets
tr_x_processed = preprocess_images(tr_x)
tr_y_repeated = torch.tensor(tr_y, dtype=torch.long).repeat_interleave(25)
train_dataset = TensorDataset(tr_x_processed.view(-1, 3, 224, 224), tr_y_repeated)  # Reshape and create dataset
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)

te_x_processed = preprocess_images(te_x)
te_y_repeated = torch.tensor(te_y, dtype=torch.long).repeat_interleave(25)
test_dataset = TensorDataset(te_x_processed.view(-1, 3, 224, 224), te_y_repeated)  # Reshape and create dataset
test_dataloader = DataLoader(test_dataset, batch_size=200, shuffle=False)

def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.logits, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct / total

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
model.train()
for epoch in range(3):  # number of epochs
    for batch in train_dataloader:
        pixel_values, labels = batch
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        total_accuracy = 0
        for batch in test_dataloader:
            pixel_values, labels = batch
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=pixel_values)
            accuracy = calculate_accuracy(outputs, labels)
            total_accuracy += accuracy

        avg_accuracy = total_accuracy / len(test_dataloader)
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Test Accuracy: {avg_accuracy * 100:.2f}%")
    model.train()


# Freeze all layers in the model
for param in model.parameters():
    param.requires_grad = True

# Unfreeze the classifier layer
for param in model.classifier.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6)


'''
