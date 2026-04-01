import torch
import torch.nn as nn
from torchvision import models

class GANLoss(nn.Module):
    def __init__(self, device):
        super(GANLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.vgg = nn.Sequential(*list(vgg.children())[:16]).to(device).eval()
        for param in self.vgg.parameters(): param.requires_grad = False

        self.adversarial_loss = nn.BCEWithLogitsLoss()

    def forward_G(self, fake, real, pred_fake):
        adv_loss = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake))
        l1 = torch.mean(torch.abs(fake - real))

        fake_vgg = (fake + 1.0) / 2.0
        real_vgg = (real + 1.0) / 2.0
        percep = torch.mean((self.vgg(fake_vgg) - self.vgg(real_vgg))**2)

        return adv_loss + (10 * l1) + (1.0 * percep)

    def forward_D(self, pred_real, pred_fake):
        real_loss = self.adversarial_loss(pred_real, torch.ones_like(pred_real))
        fake_loss = self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
        return (real_loss + fake_loss) * 0.5