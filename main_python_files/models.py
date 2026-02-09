import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DetrForObjectDetection

# DCENet architecture to implement the Zero-DCE framework
class DCENet(nn.Module):
    def __init__(self):
        super(DCENet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 3, 3, 1, 1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        A = torch.tanh(self.conv3(x2))
        enhanced = x + A * x * (1 - x)
        return enhanced, A

class JointModel(nn.Module):
    def __init__(self, detector_model="facebook/detr-resnet-50"):
        super(JointModel, self).__init__()
        # You bring BOTH models into one house
        self.enhancer = DCENet()
        self.detector = DetrForObjectDetection.from_pretrained(
            detector_model, num_labels=3, ignore_mismatched_sizes=True
        )

    ## The 'Join' happens here: 
    # The output of one (enhanced_image) becomes the input of the next
    def forward(self, pixel_values, labels=None):
        enhanced_image, A = self.enhancer(pixel_values)
        # Check if we are in training (labels provided) or inference
        if labels is not None:
            outputs = self.detector(pixel_values=enhanced_image, labels=labels)
        else:
            outputs = self.detector(pixel_values=enhanced_image)
        return outputs, enhanced_image, A