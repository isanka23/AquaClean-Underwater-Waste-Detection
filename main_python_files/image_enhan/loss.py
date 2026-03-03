import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class L_perceptual(nn.Module):
    def __init__(self):
        super(L_perceptual, self).__init__()
        # Use VGG16 to evaluate 'style' and 'texture' like the paper
        vgg = models.vgg16(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:16]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, enh, gt):
        # Calculates the difference in high-level features
        return F.mse_loss(self.feature_extractor(enh), self.feature_extractor(gt))

class L_color(nn.Module):
    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mg - mb, 2)
        return torch.pow(torch.pow(Drg, 2) + torch.pow(Drb, 2) + torch.pow(Dgb, 2), 0.5)

class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
    def forward(self, org, enh):
        org_mean = torch.mean(org, 1, keepdim=True)
        enh_mean = torch.mean(enh, 1, keepdim=True)
        org_grad = F.conv2d(org_mean, self.weight, padding=1)
        enh_grad = F.conv2d(enh_mean, self.weight, padding=1)
        return torch.mean(torch.pow(org_grad - enh_grad, 2))

class UnderwaterLosses(nn.Module):
    def __init__(self):
        super(UnderwaterLosses, self).__init__()
        self.color_loss = L_color()
        self.spa_loss = L_spa()
        self.perceptual_loss = L_perceptual() 

    def forward(self, org, enh, gt):
        # Combined objective function
        l_per = self.perceptual_loss(enh, gt)
        l_spa = self.spa_loss(org, enh)
        l_col = self.color_loss(enh)
        return 1.0 * l_per + 60 * l_spa + 5 * l_col