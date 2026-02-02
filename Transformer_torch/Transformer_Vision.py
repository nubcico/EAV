import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn as nn
import numpy as np


def calculate_accuracy(outputs, labels):
    _, predicted = torch.max(outputs.logits, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


class ImageClassifierTrainer:
    def __init__(self, DATA, model_path, sub = '', num_labels=5, lr=5e-5, batch_size=128):
        self.tr_x, self.tr_y, self.te_x, self.te_y = DATA
        self.model_path = model_path
        self.num_labels = num_labels
        self.initial_lr = lr  # Storing initial learning rate for reference
        self.batch_size = batch_size
        self.frame_per_sample = np.shape(self.tr_x)[1]  # Assuming tr_x is a numpy array or similar
        self.sub = sub  # this is for log data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.test_prediction = list()

        # Initialize model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.model.classifier = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.model.num_labels = self.num_labels

        self.model.to(self.device)

        # Initial optimizer setup with initial learning rate
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.initial_lr)

        # Prepare dataloaders, test doesn't need to be in dataloader
        print("Image preprocessing..")
        self.train_dataloader = self._prepare_dataloader(self.tr_x, self.tr_y, shuffle=True)[0]
        self.test_dataloader = self._prepare_dataloader(self.te_x, self.te_y, shuffle=False)[0]
        print("Ended..")

    def _prepare_dataloader(self, x, y, shuffle=True):
        processed_x = self.preprocess_images(x)
        y_repeated = torch.from_numpy(np.repeat(y, self.frame_per_sample)).long()

        dataset = TensorDataset(processed_x.view(-1, 3, 224, 224), y_repeated)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return dataloader, processed_x, y_repeated

    def preprocess_images(self, image_list):
        pixel_values_list = []
        for img_set in image_list:
            for img in img_set:
                processed = self.processor(images=img, return_tensors="pt")
                pixel_values = processed.pixel_values.squeeze()
                pixel_values_list.append(pixel_values)
        return torch.stack(pixel_values_list).to(self.device)

    def train(self, epochs=3, lr=None, freeze=True, log = False):
        # Update learning rate if provided, otherwise use the initial learning rate
        lr = lr if lr is not None else self.initial_lr
        if lr is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        # Freeze or unfreeze model parameters based on the freeze flag
        for param in self.model.parameters():
            param.requires_grad = not freeze
        for param in self.model.classifier.parameters():
            param.requires_grad = True

        print(f"Training with {'frozen' if freeze else 'unfrozen'} feature layers at lr={lr}")

        # Wrap the model with DataParallel
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        for epoch in range(epochs):
            # Training loop
            self.model.train()
            total_batches = len(self.train_dataloader)
            for batch_idx, batch in enumerate(self.train_dataloader, start=1):
                pixel_values, labels = [b.to(self.device) for b in batch]
                self.optimizer.zero_grad()

                outputs = self.model(pixel_values=pixel_values, labels=labels)

                if outputs.loss.dim() > 0:
                    loss = outputs.loss.mean()
                else:
                    loss = outputs.loss

                loss.backward()
                self.optimizer.step()

                print(f'batch ({batch_idx}/{total_batches})')

            
            self.model.eval()
            total_accuracy = 0
            outputs_batch = []
            with torch.no_grad():
                for batch in self.test_dataloader:
                    pixel_values, labels = [b.to(self.device) for b in batch]
                    outputs = self.model(pixel_values)

                    accuracy = calculate_accuracy(outputs, labels)
                    total_accuracy += accuracy

                    logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                    logits_cpu = logits.detach().cpu().numpy()
                    outputs_batch.append(logits_cpu)

            if epoch == epochs-1 and not freeze: # we saved test prediction only at last epoch, and finetuning
                self.outputs_test = np.concatenate(outputs_batch, axis=0)
            outputs_batch = []

            avg_accuracy = total_accuracy / len(self.test_dataloader)
            print(f"Epoch {epoch + 1}, Test Accuracy: {avg_accuracy * 100:.2f}%")

            if log:
                with open('training_performance.txt', 'a') as f:
                    f.write(f"{self.sub}, Epoch {epoch + 1}, Test Accuracy: {avg_accuracy * 100:.2f}%\n")


if __name__ == '__main__':
    import pickle
    import os
    test_acc = []
    for idx in range(1, 43):

        file_name = f"subject_{idx:02d}_vis.pkl"
        file_ = os.path.join(os.getcwd(), 'Feature_vision', file_name)

        with open(file_, 'rb') as f:
            vis_list2 = pickle.load(f)
        tr_x_vis, tr_y_vis, te_x_vis, te_y_vis = vis_list2

        mod_path = os.path.join(os.getcwd(), 'facial_emotions_image_detection')
        data = [tr_x_vis, tr_y_vis, te_x_vis, te_y_vis]
        trainer = ImageClassifierTrainer(data,
                                         model_path=mod_path, sub = f"subject_{idx:02d}",
                                         num_labels=5, lr=5e-5, batch_size=128)

        trainer.train(epochs=10, lr=5e-4, freeze=True)
        trainer.train(epochs=5, lr=5e-6, freeze=False)

        test_acc.append(trainer.outputs_test)

    import pickle
    with open("test_acc_vision.pkl", 'wb') as f:
        pickle.dump(test_acc, f)


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
        aa2 = np.reshape(aa, (200, 25, 5), 'C')
        aa3 = np.mean(aa2, 1)
        out1 = np.argmax(aa3, axis = 1)
        accuracy = np.mean(out1 == te_y_vis)
        test_acc_all.append(accuracy)

        f1 = f1_score(te_y_vis, out1, average='weighted')
        test_f1_all.append(f1)

    test_acc_all = np.reshape(np.array(test_acc_all), (42, 1))
    test_f1_all = np.reshape(np.array(test_f1_all), (42, 1))

