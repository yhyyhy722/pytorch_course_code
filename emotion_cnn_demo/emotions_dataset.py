import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2 as cv

emotion_labels = ["neutral","anger","disdain","disgust","fear","happy","sadness","surprise"]


class EmotionDataset(Dataset):
    def __init__(self, root_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        img_files = os.listdir(root_dir)
        nums_ = len(img_files)
        self.vehicle_types = []
        self.emotions = []
        self.images = []
        index = 0
        for file_name in img_files:
            emotion_attrs = file_name.split("_")
            emotion_ = np.int32(emotion_attrs[0])
            self.images.append(os.path.join(root_dir, file_name))
            self.emotions.append(emotion_)
            index += 1

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
        img = cv.resize(img, (64, 64))
        img = (np.float32(img) /255.0 - 0.5) / 0.5
        # H, W C to C, H, W
        img = img.transpose((2, 0, 1))
        sample = {'image': torch.from_numpy(img), 'emotion': self.emotions[idx]}
        return sample


if __name__ == "__main__":
    ds = EmotionDataset("D:/facedb/emotion_dataset")
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample['image'].size(), sample['emotion'])
        if i == 3:
            break

    dataloader = DataLoader(ds, batch_size=4, shuffle=True)
    # data loader
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['emotion'])
        break