from Dataload_eeg import *
import torch
import torch.nn as nn
from Fusion.VIT_audio.Transformer_audio import Trainer_uni
#from tensorflow.keras.models import Model
from torch.nn.utils import weight_norm

# Define the EEGNet model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm


class EEGNet_tor(nn.Module):
    def __init__(self, nb_classes, Chans=30, Samples=500, dropoutRate=0.5, kernLength=300, F1=8, D=8, F2=64,
                 norm_rate=1.0, dropoutType='Dropout'):
        super(EEGNet_tor, self).__init__()

        # Configure dropout
        self.dropout = nn.Dropout(dropoutRate) if dropoutType == 'Dropout' else nn.Dropout2d(dropoutRate)

        # Block 1
        self.firstConv = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.firstBN = nn.BatchNorm2d(F1)
        self.elu = nn.ELU()

        self.depthwiseConv = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, padding=0, bias=False)
        self.depthwiseBN = nn.BatchNorm2d(F1 * D)
        self.depthwisePool = nn.AvgPool2d((1, 4))

        # Applying max-norm constraint
        self.depthwiseConv.register_forward_hook(
            lambda module, inputs, outputs: module.weight.data.renorm_(p=2, dim=0, maxnorm=norm_rate))

        # Block 2
        self.separableConv = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.separableBN = nn.BatchNorm2d(F2)
        self.separablePool = nn.AvgPool2d((1, 8))

        # Final layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(F2 * ((Samples // 4 // 8)), nb_classes)
        self.softmax = nn.Softmax(dim=1)

        # Applying max-norm constraint
        self.dense.register_forward_hook(
            lambda module, inputs, outputs: module.weight.data.renorm_(p=2, dim=0, maxnorm=norm_rate))

    def forward(self, x):
        x = self.firstConv(x)
        x = self.firstBN(x)
        x = self.elu(x)
        x = self.depthwiseConv(x)
        x = self.depthwiseBN(x)
        x = self.elu(x)
        x = self.depthwisePool(x)
        x = self.dropout(x)
        x = self.separableConv(x)
        x = self.separableBN(x)
        x = self.elu(x)
        x = self.separablePool(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x



def _prepare_dataloader(self, x, y, shuffle=False):
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    return dataloader


if __name__ == "__main__":
    result_acc = list()
    for i in range(1, 43):
        eeg_loader = DataLoadEEG(subject=i, band=[5, 30], fs_orig=500, fs_target=100,
                                 parent_directory=r'C:\Users\minho.lee\Dropbox\Datasets\EAV')
        data_eeg, data_eeg_y = eeg_loader.data_prepare()

        division_eeg = EAVDataSplit(data_eeg, data_eeg_y)
        [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg] = division_eeg.get_split()
        tr_x_eeg = torch.from_numpy(tr_x_eeg).float().unsqueeze(1)  # Reshape to (200, 1, 30, 500)
        te_x_eeg = torch.from_numpy(te_x_eeg).float().unsqueeze(1)  # Reshape to (200, 1, 30, 500)

        data = [tr_x_eeg, tr_y_eeg, te_x_eeg, te_y_eeg]

        #####################################################
        model = EEGNet_tor(nb_classes=5, D=8, F2=64, Chans=30, kernLength=300, Samples=500,
                       dropoutRate=0.5)
        trainer = Trainer_uni(model=model, data=data, lr=1e-5, batch_size=32, num_epochs=350)
        trainer.train()
        model.eval()  # 모델을 평가 모드로 설정

        # 손실 함수 설정
        criterion = nn.CrossEntropyLoss()
        # 데이터를 PyTorch 텐서로 변환 및 GPU로 이동
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        te_x_eeg = torch.tensor(te_x_eeg, dtype=torch.float32).to(device)
        te_y_eeg = torch.tensor(te_y_eeg, dtype=torch.long).to(device)
        model.to(device)
        # 평가 실행
        with torch.no_grad():
            scores = model(te_x_eeg)
            predictions = scores.argmax(dim=1)
            correct = (predictions == te_y_eeg).sum().item()
            total = te_y_eeg.size(0)  # 전체 데이터 개수
            accuracy = correct / total

        result_acc.append(accuracy)
        print(result_acc)


