#coding=utf-8
import cv2
import torch
from torch.utils.data import Dataset
import os
import glob

class MnistTrainDataset(Dataset):
    def __init__(self, root):
        super(MnistTrainDataset, self).__init__()
        self.root = root
        self.imgs = []
        self.labels = []

        for file in os.listdir(os.path.join(self.root , "train")):
            new_images = glob.glob(os.path.join(self.root, "train", file, "*.jpg"))
            self.imgs.extend(new_images)
            self.labels.extend([int(file) for _ in range(len(new_images))])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 不做 Normalize
        img = torch.tensor(cv2.imread(self.imgs[idx], cv2.IMREAD_GRAYSCALE), dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
    
class MnistValDataset(Dataset):
    def __init__(self, root):
        super(MnistValDataset, self).__init__()
        self.root = root
        self.imgs = []
        self.labels = []

        for file in os.listdir(os.path.join(self.root , "val")):
            new_images = glob.glob(os.path.join(self.root, "val", file, "*.jpg"))
            self.imgs.extend(new_images)
            self.labels.extend([int(file) for _ in range(len(new_images))])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 不做 Normalize
        img = torch.tensor(cv2.imread(self.imgs[idx], cv2.IMREAD_GRAYSCALE), dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

class MnistTestDataset(Dataset):
    def __init__(self, root):
        super(MnistTestDataset, self).__init__()
        self.root = root
        self.imgs = []

        new_images = glob.glob(os.path.join(self.root, "test", "*.jpg"))
        self.imgs.extend(new_images)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # 不做 Normalize
        img = torch.tensor(cv2.imread(self.imgs[idx], cv2.IMREAD_GRAYSCALE), dtype=torch.float32).unsqueeze(0)
        return img