import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import cv2 as cv

color_labels = ["white", "gray", "yellow", "red", "green", "blue", "black"]
type_labels = ["car", "bus", "truck", "van"]


class VehicleAttrsDataset(Dataset):
    def __init__(self, root_dir):
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                          std=[0.5, 0.5, 0.5]),
                                             transforms.Resize((72, 72))
                                            ])
        img_files = os.listdir(root_dir)
        nums_ = len(img_files)
        # vehicle type:  [car:0, bus:1, truck:2, van:3]
        # vehicle color:[white:0, gray:1, yellow:2, red:3, green:4, blue:5, black:6]
        self.vehicle_types = []
        self.vehicle_colors = []
        self.images = []
        index = 0
        for file_name in img_files:
            color_type_group = file_name.split("_")
            color_ = color_labels.index(color_type_group[0])
            type_ = type_labels.index(color_type_group[1])
            self.vehicle_colors.append(np.float32(color_))
            self.vehicle_types.append(np.float32(type_))
            self.images.append(os.path.join(root_dir, file_name))
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
        sample = {'image': self.transform(img), 'color': self.vehicle_colors[idx], 'type': self.vehicle_types[idx]}
        return sample


if __name__ == "__main__":
    ds = VehicleAttrsDataset("D:/facedb/vehicle_attrs_dataset")
    for i in range(len(ds)):
        sample = ds[i]
        print(i, sample['image'].size(), sample['color'])
        if i == 3:
            break

    dataloader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4)
    # data loader
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(), sample_batched['type'])
        break
    # src_dir = "D:/vehicle_attrs_dataset"
    # dst_dir = "D:/facedb/vehicle_attrs_dataset"
    # files = os.listdir(src_dir)
    # for f in files:
    #     image = cv.imread(os.path.join(src_dir, f))
    #     dst = cv.resize(image, (72, 72))
    #     cv.imwrite(os.path.join(dst_dir, f), dst)
    #     print(f, "havd done!")