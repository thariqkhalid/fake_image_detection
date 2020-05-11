from data_loader import load_split_train_test

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F






DATA_DIR = "D:/Madiha Mariam Ahmed/Image Forgery Detection/phase-01-training/dataset-dist/phase-01/training/"



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            #nn.BatchNorm2d(16), # Used to improve speed, performance and stability of NN. Also, is BatchNorm2d() important?
            nn.ReLU()
            nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            #nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(32,64, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))



        self.fc = nn.Linear( 3.125, 2)

    def forward(self, x):
        out = self.layer1(x)
        print(out.shape)

        out = self.layer2(out)
        print(out.shape)

        out = self.layer3(out)
        print(out.shape)

        out = self.layer4(out)
        print(out.shape)

        out = self.layer5(out)
        print(out.shape)

        out = out.reshape(out.size (0), -1) # Flatten
        print(out.shape)
        return out # 'out' will give the size of the resulting feature map. Sir, can you please run it and check?









def train_cnn(train_data):

    trainloader, testloader = load_split_train_test(DATA_DIR, .2)

    for epoch in range(2):

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):

            inputs, labels = data

            optimizer.zero_grad() # Backprop

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 2000 == 1999: # For every 2000 steps print loss

                print('[%d, %5d] loss: %.3f' %

                      (epoch + 1, i + 1, running_loss / 2000))

                running_loss = 0.0





    print('Finished Training')

    PATH = './'

    torch.save(net.state_dict(), PATH)



if __name__ == '__main__':

    model = ConvNet()

    criterion = nn.CrossEntropyLoss() # For classification probs

    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainloader, testloader = load_split_train_test(DATA_DIR, .2)



    train_results = train_cnn(trainloader)