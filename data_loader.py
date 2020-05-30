import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

# THERE ARE MULTIPLE WAYS TO HAVE CUSTOM DATA LOADERS. WE'LL EXPLORE BOTH AND SEE WHICH IS GOOD FOR US

DATA_DIR = "D:/Madiha Mariam Ahmed/Image Forgery Detection/phase-01-training/dataset-dist/phase-01/training/"
'''
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
'''


def load_split_train_test(data_dir, valid_size=0.2):
    # first let's create a dataloader with which we can calculate the mean and standard deviation for the whole data
    data_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    full_data = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    data_sampler = SubsetRandomSampler(full_data)
    dataloader = torch.utils.data.DataLoader(full_data, sampler=data_sampler, batch_size=4)
    mean_list = []
    std_list = []
    for i, data in enumerate(dataloader, 0):
        numpy_img = data[0].numpy()
        batch_mean = np.mean(numpy_img, axis=(0, 2, 3))
        batch_std = np.std(numpy_img, axis=(0, 2, 3))
        mean_list.append(batch_mean)
        std_list.append(batch_std)
    mean_list = np.array(mean_list).mean()
    std_list = np.array(std_list).mean()
    train_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean_list, std_list)])
    test_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean_list, std_list)])
    train_data = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
    test_data = datasets.ImageFolder(DATA_DIR, transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=4)
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=4)
    return trainloader, testloader

if __name__ == '__main__':
    trainloader, testloader = load_split_train_test(DATA_DIR, .2)
    print(trainloader.dataset.classes)
