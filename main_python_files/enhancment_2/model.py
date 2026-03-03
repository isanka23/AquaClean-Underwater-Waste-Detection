import torch
import torch.nn as nn

# --- NOVELTY: SPATIAL ATTENTION BLOCK ---
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Large kernel to capture spatial context around objects
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise max and average pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate and convolve to create the attention map
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        # Weight the input features by the attention map
        return x * self.sigmoid(out)

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        def upconv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.e1 = nn.Conv2d(3, 64, 4, 2, 1) 
        self.e2 = conv_block(64, 128)
        self.e3 = conv_block(128, 256)
        self.e4 = conv_block(256, 512)

        # ATTENTION BOTTLENECK
        self.attention = SpatialAttention()

        # Decoder
        self.d1 = upconv_block(512, 256)
        self.d2 = upconv_block(512, 128) 
        self.d3 = upconv_block(256, 64)  
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, 4, 2, 1), 
            nn.Tanh() # Scales to [-1, 1] for normalization consistency
        )

    def forward(self, x):
        enc1 = self.e1(x)
        enc2 = self.e2(enc1)
        enc3 = self.e3(enc2)
        enc4 = self.e4(enc3)

        # Apply spatial attention to focus on structural debris features
        focused = self.attention(enc4)

        dec1 = self.d1(focused)
        dec2 = self.d2(torch.cat([dec1, enc3], 1))
        dec3 = self.d3(torch.cat([dec2, enc2], 1))
        
        return self.final(torch.cat([dec3, enc1], 1))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, stride=2):
            return nn.Sequential(
                nn.Conv2d(in_filters, out_filters, 4, stride, 1),
                nn.BatchNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True)
            )
        # PatchGAN approach: concat murky input and enhanced result
        self.model = nn.Sequential(
            discriminator_block(6, 64, stride=2),
            discriminator_block(64, 128, stride=2),
            discriminator_block(128, 256, stride=2),
            discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, img_A, img_B):
        return self.model(torch.cat((img_A, img_B), 1))