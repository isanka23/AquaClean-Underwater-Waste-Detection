import torch
import torch.nn as nn


# ==========================================================
# CBAM ATTENTION MODULE
# This module helps the generator focus on important features.
# ==========================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        # Pooling is used to summarize each channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared small network to learn channel importance
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Use both average and max pooled features
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))

        # Generate channel attention map and apply it to input features
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = 3 if kernel_size == 7 else 1

        # Takes average and max maps, then learns spatial importance
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Create spatial descriptors from channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Combine descriptors and apply spatial attention
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv1(x_cat))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()

        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # CBAM applies channel attention first, then spatial attention
        out = self.ca(x)
        out = self.sa(out)
        return out


# ==========================================================
# 5-LAYER U-NET GENERATOR
# This generator takes a degraded underwater image and outputs
# an enhanced image.
#
# Encoder  : extracts features from the input image.
# Bottleneck + CBAM : refines deep features using attention.
# Decoder  : reconstructs the enhanced image.
# Skip connections preserve low-level details.
# ==========================================================

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(UNetGenerator, self).__init__()

        # Encoder block: downsamples the image and extracts features
        def conv_block(in_ch, out_ch, normalize=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1)]

            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Decoder block: upsamples feature maps to reconstruct image
        def dec_block(in_ch, out_ch, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]

            if dropout:
                layers.append(nn.Dropout(0.5))

            return nn.Sequential(*layers)

        # Encoder path
        self.e1 = conv_block(in_channels, features, normalize=False)
        self.e2 = conv_block(features, features * 2)
        self.e3 = conv_block(features * 2, features * 4)
        self.e4 = conv_block(features * 4, features * 8)
        self.e5 = conv_block(features * 8, features * 8)

        # Attention is applied at the bottleneck
        self.attention = CBAM(features * 8)

        # Decoder path
        self.d1 = dec_block(features * 8, features * 8, dropout=True)
        self.d2 = dec_block(features * 8 * 2, features * 4, dropout=True)
        self.d3 = dec_block(features * 4 * 2, features * 2)
        self.d4 = dec_block(features * 2 * 2, features)

        # Final layer outputs a 3-channel enhanced RGB image
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features * 2, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder feature extraction
        enc1 = self.e1(x)
        enc2 = self.e2(enc1)
        enc3 = self.e3(enc2)
        enc4 = self.e4(enc3)
        enc5 = self.e5(enc4)

        # Bottleneck attention refinement
        focused = self.attention(enc5)

        # Decoder reconstruction with skip connections
        dec1 = self.d1(focused)
        dec2 = self.d2(torch.cat([dec1, enc4], 1))
        dec3 = self.d3(torch.cat([dec2, enc3], 1))
        dec4 = self.d4(torch.cat([dec3, enc2], 1))

        # Final output with last skip connection
        return self.final(torch.cat([dec4, enc1], 1))


# ==========================================================
# PATCHGAN DISCRIMINATOR
# This discriminator checks whether the enhanced image looks real.
# It receives both input image and target/generated image together.
# PatchGAN focuses on local texture and style instead of only
# judging the whole image globally.
# ==========================================================

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride, 1)]

            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))

            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # Input has 6 channels because two RGB images are concatenated:
        # input image + real/enhanced image
        self.model = nn.Sequential(
            discriminator_block(in_channels * 2, 64, normalize=False),
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img_a, img_b):
        # Concatenate input and output image before discrimination
        img_input = torch.cat((img_a, img_b), 1)
        return self.model(img_input)