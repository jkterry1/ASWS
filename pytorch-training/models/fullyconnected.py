import torch
import torch.nn as nn

class FC1(nn.Module):

    def __init__(self):
        super(FC1, self).__init__()
        # input layer has 3072 features
        # first hidden layer has 400 features
        # second hidden layer has 300 features
        # output should be 10
        self.classifier = nn.Sequential(nn.Linear(3072, 400), nn.Linear(400, 300), nn.Linear(300, 10))

    def forward(self, x):
        ###
        # CIFAR10 is a 3 channel 32x32 image
        # flatten yields 3072 dimensional vector
        ##
        x = x.view(-1, 3072)
        out = self.classifier(x)
        return out


class FC2(nn.Module):

    def __init__(self):
        super(FC2, self).__init__()
        # input layer has 3072 features
        # first hidden layer has 700 features
        # second hidden layer has 600 features
        # third hidden layer has 500 features
        # output should be 10
        self.classifier = nn.Sequential(nn.Linear(3072, 700), nn.Linear(700, 600), nn.Linear(600, 500), nn.Linear(500, 10))

    def forward(self, x):
        x = x.view(-1, 3072)
        out = self.classifier(x)
        return out