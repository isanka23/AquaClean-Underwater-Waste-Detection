import torch
import torch.nn as nn

class AquaCleanNet(nn.Module):
    def __init__(self, nf=32):
        super(AquaCleanNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(nf*2, 24, 3, 1, 1, bias=True) # 24 maps = 3 channels * 8 iterations
    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        # Estimating pixel-wise adjustment curves [cite: 47, 90]
        curves = self.e_conv7(torch.cat([x1, x6], 1))
        r_list = torch.split(curves, 3, dim=1)
        y = x
        for r in r_list:
            y = y + r * (torch.pow(y, 2) - y)
        return y