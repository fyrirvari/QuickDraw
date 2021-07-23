import numpy as np
from torch.utils.data import Dataset
import json


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_tensor_file, transform=None):
        self.transform = transform
        with open(annotations_file, 'r') as file:
            file.seek(0)
            self.img_labels = np.array(json.load(file))
        with open(img_tensor_file, 'r') as file:
            file.seek(0)
            self.img_tensor = np.array(json.load(file))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.img_tensor[idx]
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, int(label)
