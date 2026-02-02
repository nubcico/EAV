import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class EAVDataSplit:
    def __init__(self, x, y, batch_size=32):
        self.x = np.array(x)
        self.y = np.array(y)
        self.batch_size = batch_size
    def _split_features_labels(self):
        # Splitting features and labels based on class, select each 80 samples per class in order
        features = []
        labels = []
        for class_idx in range(5):  # Assuming there are 5 classes
            class_mask = np.where(self.y == class_idx)
            class_features = self.x[class_mask]
            class_labels = self.y[class_mask]

            features.append(class_features)
            labels.append(class_labels)

        return features, labels

    def get_split(self, h_idx=40): # update it if you want to use different ratio here we have 50/50
        [features, labels] = self._split_features_labels()
        # Splitting into training and testing
        train_features = np.concatenate([cls_features[:h_idx] for cls_features in features], axis=0)
        test_features = np.concatenate([cls_features[h_idx:] for cls_features in features], axis=0)
        train_labels = np.concatenate([cls_labels[:h_idx] for cls_labels in labels], axis=0)
        test_labels = np.concatenate([cls_labels[h_idx:] for cls_labels in labels], axis=0)

        #
        train_features = np.squeeze(train_features)
        test_features = np.squeeze(test_features)
        train_labels = train_labels
        test_labels = test_labels

        return train_features, train_labels, test_features, test_labels

    def get_loaders(self):
        self._split_features_labels()
        train_features, train_labels, test_features, test_labels = self.get_split()

        train_features = torch.Tensor(np.squeeze(train_features))
        test_features = torch.Tensor(np.squeeze(test_features))
        train_labels = torch.Tensor(train_labels).long()  # Using .long() for labels
        test_labels = torch.Tensor(test_labels).long()

        # Creating TensorDatasets and DataLoaders
        train_dataset = TensorDataset(train_features, train_labels)
        test_dataset = TensorDataset(test_features, test_labels)

        loader_train = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        loader_test = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return loader_train, loader_test

'''
# 50/50 in sequential order, straight forward codes without loops
def EAV_train_test(x, y):
    c1 = np.where(np.array(y) == 0)
    c2 = np.where(np.array(y) == 1)
    c3 = np.where(np.array(y) == 2)
    c4 = np.where(np.array(y) == 3)
    c5 = np.where(np.array(y) == 4)
    # (400, 1, 1024, 128), np.array(feature)
    f1 = np.array(x)[c1]
    f2 = np.array(x)[c2]
    f3 = np.array(x)[c3]
    f4 = np.array(x)[c4]
    f5 = np.array(x)[c5]

    h_idx = 40
    train_features = np.concatenate((f1[:h_idx], f2[:h_idx], f3[:h_idx], f4[:h_idx], f5[:h_idx]), axis=0)
    test_features = np.concatenate((f1[h_idx:], f2[h_idx:], f3[h_idx:], f4[h_idx:], f5[h_idx:]), axis=0)
    train_features = torch.Tensor(np.squeeze(train_features))
    test_features = torch.Tensor(np.squeeze(test_features))

    # do this to the label as well
    c1 = np.where(np.array(y) == 0)
    c2 = np.where(np.array(y) == 1)
    c3 = np.where(np.array(y) == 2)
    c4 = np.where(np.array(y) == 3)
    c5 = np.where(np.array(y) == 4)
    # (400, 1, 1024, 128), np.array(feature)
    l1 = np.array(y)[c1]
    l2 = np.array(y)[c2]
    l3 = np.array(y)[c3]
    l4 = np.array(y)[c4]
    l5 = np.array(y)[c5]

    train_labels = np.concatenate((l1[:h_idx], l2[:h_idx], l3[:h_idx], l4[:h_idx], l5[:h_idx]), axis=0)
    test_labels = np.concatenate((l1[h_idx:], l2[h_idx:], l3[h_idx:], l4[h_idx:], l5[h_idx:]), axis=0)
    train_labels = torch.Tensor(train_labels)
    test_labels = torch.Tensor(test_labels)

    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    loader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    loader_test = DataLoader(test_dataset, batch_size=32, shuffle=True)
    return loader_train, loader_test
'''

