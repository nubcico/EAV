import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn as nn
import numpy as np
import pickle
import cv2
from torchvision import transforms
import torch.nn.functional as F
from torchvision.models import resnet50
from PIL import Image
import ipdb

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


class VideoModel(nn.Module):
    def __init__(self, num_labels=5, ratio=1):
        self.num_labels = num_labels
        super(VideoModel, self).__init__()
        base_model = resnet50(pretrained=True, progress=True)
        self.base_model = torch.nn.Sequential(*(list(base_model.children())[:-2]))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_layer_one = nn.Linear(2048 // ratio, 2048)
        self.shared_layer_two = nn.Linear(2048, 2048)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, self.num_labels)
        self.ratio = ratio

    def channel_attention(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_pool = self.shared_layer_one(avg_pool.view(avg_pool.size(0), -1))
        max_pool = self.shared_layer_one(max_pool.view(max_pool.size(0), -1))
        avg_pool = self.shared_layer_two(avg_pool)
        max_pool = self.shared_layer_two(max_pool)
        return avg_pool + max_pool

    def forward(self, x):
        x = self.base_model(x)
        #ipdb.set_trace()
        attention = self.channel_attention(x)
        x = x * attention.unsqueeze(2).unsqueeze(3)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class ImageClassifierTrainer:
    def __init__(self, DATA, num_labels=5, lr=5e-5, batch_size=128):
        self.tr_x, self.tr_y, self.te_x, self.te_y = DATA
        self.num_labels = num_labels
        self.initial_lr = lr  # Storing initial learning rate for reference
        self.batch_size = batch_size
        self.frame_per_sample = np.shape(self.tr_x)[1]  # Assuming tr_x is a numpy array or similar

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model and processor
        self.processor=transform
        self.model = VideoModel(self.num_labels)
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        # Initial optimizer setup with initial learning rate
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.initial_lr)

        # Prepare dataloaders
        print("Image preprocessing..")
        self.train_dataloader = self._prepare_dataloader(self.tr_x, self.tr_y)
        self.test_dataloader = self._prepare_dataloader(self.te_x, self.te_y, shuffle=False)
        print("Ended..")
    def calculate_accuracy(self,outputs, labels):
        #ipdb.set_trace()
        #probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        return correct / labels.size(0)
    def _prepare_dataloader(self, x, y, shuffle=True):
        processed_x = self.preprocess_images(x)
        y_repeated = torch.from_numpy(np.repeat(y, self.frame_per_sample)).long()

        dataset = TensorDataset(processed_x.view(-1, 3, 224, 224), y_repeated)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        del dataset
        return dataloader
    def _delete_dataloader(self):
        del self.train_dataloader
        del self.test_dataloader

    def preprocess_images(self, image_list):
        pixel_values_list = []
        for img_set in image_list:
            for img in img_set:
                pil_img = Image.fromarray(img)
                processed = self.processor(pil_img)
                #ipdb.set_trace()
                pixel_values = processed.squeeze() #check if the shape is (_, 224, 224, 3)
                pixel_values_list.append(pixel_values)
        return torch.stack(pixel_values_list).to(self.device)

    def train(self, epochs=3, lr=None, freeze=True):
        # Update learning rate if provided, otherwise use the initial learning rate
        lr = lr if lr is not None else self.initial_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # Freeze or unfreeze model parameters based on the freeze flag
        for param in self.model.base_model.parameters():
            param.requires_grad = not freeze

        print(f"Training with {'frozen' if freeze else 'unfrozen'} feature layers at lr={lr}")

        # Wrap the model with DataParallel
        #if torch.cuda.device_count() > 1:
        #    self.model = nn.DataParallel(self.model)

        for epoch in range(epochs):
            # Training loop
            self.model.train()
            total_accuracy_train = 0
            outputs_batch = []
            for batch in self.train_dataloader:
                pixel_values, labels = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()
                #ipdb.set_trace()
                outputs = self.model(pixel_values)
                loss=self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                accuracy_train = self.calculate_accuracy(outputs, labels)
                total_accuracy_train += accuracy_train
            # Evaluation loop
            avg_accuracy_train = total_accuracy_train / len(self.train_dataloader)
            self.model.eval()
            total_accuracy = 0
            with torch.no_grad():
                for batch in self.test_dataloader:
                    pixel_values, labels = [b.to(self.device) for b in batch]
                    outputs = self.model(pixel_values)
                   

                    accuracy = self.calculate_accuracy(outputs, labels)
                    total_accuracy += accuracy
                    
                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    logits_cpu = logits.detach().cpu().numpy()
                    outputs_batch.append(logits_cpu)

            if epoch == epochs-1 and not freeze: # we saved test prediction only at last epoch, and finetuning
                self.outputs_test = np.concatenate(outputs_batch, axis=0)
                        
            outputs_batch = []
            avg_accuracy = total_accuracy / len(self.test_dataloader)
            print(f"Epoch {epoch + 1}, Train Accuracy: {avg_accuracy_train * 100:.2f}%, Test Accuracy: {avg_accuracy * 100:.2f}%")
            
        





# Example usage
from sklearn.metrics import f1_score
if __name__ == '__main__':

    import pickle
    import os

    test_acc_all = list()
    test_f1_all = list()
    for idx in range(1, 2):
        test_acc = []
        torch.cuda.empty_cache()
        direct=r"C:\Users\user.DESKTOP-HI4HHBR\Downloads\Feature_vision"
        file_name = f"subject_{idx:02d}_vis.pkl"
        file_ = os.path.join(direct, file_name)

        with open(file_, 'rb') as f:
            vis_list2 = pickle.load(f)
        tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = vis_list2

        mod_path = r'C:\Users\user.DESKTOP-HI4HHBR\Downloads\facial_emotions_image_detection (1)'
        data = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]
        trainer = ImageClassifierTrainer(data,num_labels=5, lr=5e-5, batch_size=32)

        trainer.train(epochs=3, lr=5e-4, freeze=True)
        trainer.train(epochs=3, lr=5e-6, freeze=False)
        trainer._delete_dataloader()
        test_acc.append(trainer.outputs_test)

        #ipdb.set_trace()
        f = open("accuracy_cnn.txt", "a")
        f.write("\n Subject ")
        f.write(str(idx))
        aa = test_acc[0]
        aa2 = np.reshape(aa, (200, 25, 5), 'C')
        aa3 = np.mean(aa2, 1)
        out1 = np.argmax(aa3, axis = 1)
        accuracy = np.mean(out1 == te_y_vis)
        test_acc_all.append(accuracy)
        f.write("\n")
        f.write(f"The accuracy of the {idx}-subject is ")
        f.write(str(accuracy))
        print(f"The accuracy of the {idx}-subject is ")
        print(accuracy)
        f1 = f1_score(te_y_vis, out1, average='weighted')
        test_f1_all.append(f1)
        f.write("\n")
        f.write(f"The f1 score of the {idx}-subject is ")
        f.write(str(f1))
        f.close()
        print(f"The f1 score of the {idx}-subject is ")
        print(f1)

    test_acc_all = np.reshape(np.array(test_acc_all), (42, 1))
    test_f1_all = np.reshape(np.array(test_f1_all), (42, 1))
