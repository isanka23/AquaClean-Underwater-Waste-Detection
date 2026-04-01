# Cell 3: Create Your Custom model.py This cell uses a special command (%%writefile) to instantly create your custom PyTorch architecture on the Colab server. It perfectly preserves your CBAM attention block, 5-layer U-Net, Instance Normalization, and Dropout.

import torch
import torch.nn as nn

# ==========================================
# 1. FULL CBAM (Channel + Spatial Attention)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv1(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.ca(x)
        out = self.sa(out)
        return out

# ==========================================
# 2. 5-LAYER U-NET GENERATOR
# ==========================================
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(UNetGenerator, self).__init__()

        def conv_block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def dec_block(in_ch, out_ch, dropout=False):
            layers = [nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                      nn.InstanceNorm2d(out_ch),
                      nn.ReLU(inplace=True)]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.e1 = conv_block(in_channels, features, normalize=False)
        self.e2 = conv_block(features, features * 2)
        self.e3 = conv_block(features * 2, features * 4)
        self.e4 = conv_block(features * 4, features * 8)
        self.e5 = conv_block(features * 8, features * 8)

        self.attention = CBAM(features * 8)

        self.d1 = dec_block(features * 8, features * 8, dropout=True)
        self.d2 = dec_block(features * 8 * 2, features * 4, dropout=True)
        self.d3 = dec_block(features * 4 * 2, features * 2)
        self.d4 = dec_block(features * 2 * 2, features)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        enc1 = self.e1(x)
        enc2 = self.e2(enc1)
        enc3 = self.e3(enc2)
        enc4 = self.e4(enc3)
        enc5 = self.e5(enc4)

        focused = self.attention(enc5)

        dec1 = self.d1(focused)
        dec2 = self.d2(torch.cat([dec1, enc4], 1))
        dec3 = self.d3(torch.cat([dec2, enc3], 1))
        dec4 = self.d4(torch.cat([dec3, enc2], 1))

        return self.final(torch.cat([dec4, enc1], 1))

# ==========================================
# 3. DISCRIMINATOR
# ==========================================
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            discriminator_block(in_channels * 2, 64, normalize=False),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img_a, img_b):
        img_input = torch.cat((img_a, img_b), 1)
        return self.model(img_input)
