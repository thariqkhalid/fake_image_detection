import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms, models

# THERE ARE MULTIPLE WAYS TO HAVE CUSTOM DATA LOADERS. WE'LL EXPLORE BOTH AND SEE WHICH IS GOOD FOR US

DATA_DIR = "dataset-dist/phase-01/training/"

class FakeImagesDataset(Dataset):
    def __init__(self):
        self.transformations = transforms.Compose([transforms.CenterCrop(100),
                                                  transforms.ToTensor()])

    def __getitem__(self, index):
        data =
        data = self.transformations(data)

        return (img, label)

    def __len__(self):
        return count


def load_split_train_test(DATA_DIR, valid_size = 0.2):
    train_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.ToTensor()])
    train_data =  datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    test_data = datasets.ImageFolder(DATA_DIR, transform=test_transforms)