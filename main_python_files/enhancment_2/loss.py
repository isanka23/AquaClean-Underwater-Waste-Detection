import torch
import torch.nn as nn
from torchvision import models

class GANLoss(nn.Module):
    def __init__(self, device):
        super(GANLoss, self).__init__()
        # Use VGG16 features for Perceptual Loss (Deep style similarity)
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.vgg = nn.Sequential(*list(vgg.children())[:16]).to(device).eval()
        for param in self.vgg.parameters(): param.requires_grad = False
        
        self.adversarial_loss = nn.BCEWithLogitsLoss() 

    def forward_G(self, fake, real, pred_fake):
        # Adversarial component (fooling the critic)
        adv_loss = self.adversarial_loss(pred_fake, torch.ones_like(pred_fake))
        # Pixel-wise Content component
        l1 = torch.mean(torch.abs(fake - real))
        # Perceptual Style component
        percep = torch.mean((self.vgg(fake) - self.vgg(real))**2)
        
        return adv_loss + (10 * l1) + (1.0 * percep)

    def forward_D(self, pred_real, pred_fake):
        loss_real = self.adversarial_loss(pred_real, torch.ones_like(pred_real))
        loss_fake = self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
        return (loss_real + loss_fake) * 0.5