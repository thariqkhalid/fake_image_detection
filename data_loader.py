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


def load_split_train_test(data_dir, valid_size = 0.2):
    train_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.Resize(224),
                                           transforms.ToTensor()])
    def show (i):
        for i in train_transforms+test_transforms:
            i = i.reshape((224, 224, 3))
            m, M = i.min(), i.max()
            plt.imshow((i - m) / (M - m))
            plt.show()
    
    train_data =  datasets.ImageFolder(data_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir, transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                                              sampler=train_sampler, batch_size=4)
    testloader = torch.utils.data.DataLoader(test_data,
                                             sampler=test_sampler, batch_size=4)
    return trainloader, testloader


trainloader, testloader = load_split_train_test(DATA_DIR, .2)
print(trainloader.dataset.classes)
