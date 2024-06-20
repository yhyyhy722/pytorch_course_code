import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2 as cv

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALL_CHAR_SET = NUMBER + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 5


def output_nums():
    return MAX_CAPTCHA * ALL_CHAR_SET_LEN


def encode(a):
    onehot = [0]*ALL_CHAR_SET_LEN
    idx = ALL_CHAR_SET.index(a)
    onehot[idx] += 1
    return onehot


class CapchaDataset(Dataset):
    def __init__(self, root_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        img_files = os.listdir(root_dir)
        self.txt_labels = []
        self.encodes = []
        self.images = []
        for file_name in img_files:
            label = file_name[:-4]
            label_oh = []
            for i in label:
                label_oh += encode(i)
            self.images.append(os.path.join(root_dir, file_name))
            self.encodes.append(np.array(label_oh))
            self.txt_labels.append(label)

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image_path = self.images[idx]
        else:
            image_path = self.images[idx]
        img = cv.imread(image_path)  # BGR order
        h, w, c = img.shape
        # rescale
        img = cv.resize(img, (128, 32))
        img = (np.float32(img) /255.0 - 0.5) / 0.5
        # H, W C to C, H, W
        img = img.transpose((2, 0, 1))
        sample = {'image': torch.from_numpy(img), 'encode': self.encodes[idx], 'label': self.txt_labels[idx]}
        return sample


if __name__ == "__main__":
    ds = CapchaDataset("D:/python/pytorch_tutorial/capcha/samples")
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample['image'].size(), sample['label'], sample['encode'].shape)
        print("标签编码", sample['encode'].reshape(5, -1))
        if i == 1:
            break
