from ImageDataset import ImageDataset
import torch


def GetLoader(data_path, label_path):
    data_set = ImageDataset(label_path, data_path)
    data_loader = torch.utils.data.DataLoader(data_set, shuffle=True)
    return data_loader
