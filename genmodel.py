import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision import models

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# PyTorch v0.4.0
model = Net().to(device)
#print(model)
# Need to provide the input layer in the folllowing format
#print("Input-0\t\t[-1, 3, 224, 224]\t 0")
#summary(model, input_size)

#modelvgg = models.vgg16().to(device)
modelvgg = models.resnet50().to(device)

input_size = (3, 224, 224)
print("\n\n\n=Shape Start=")
print("    Input-0     [-1, 3, 224, 224]     0")
summary(modelvgg, input_size)
print("=Shape End=")

print("\n\n\n=Model Start=")
print(modelvgg)
print("=Model End=")

