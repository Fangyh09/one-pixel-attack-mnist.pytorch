'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.sigmoid = nn.Sigmoid()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.sigmoid(self.avg_pool(self.conv1(x)))
        x = self.sigmoid(self.avg_pool(self.conv2_drop(self.conv2(x))))
        x = x.view(-1, 320)
        x = F.sigmoid(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return self.log_softmax(x)


# test()
