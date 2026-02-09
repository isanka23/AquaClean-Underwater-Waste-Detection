import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from transformers import DetrForObjectDetection

# ==========================================
# 1. DEFINE CLASSES (Must match your training)
# ==========================================
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
        self.enhancer = DCENet()
        self.detector = DetrForObjectDetection.from_pretrained(
            detector_model, num_labels=3, ignore_mismatched_sizes=True
        )
    def forward(self, pixel_values):
        enhanced_image, A = self.enhancer(pixel_values)
        outputs = self.detector(pixel_values=enhanced_image)
        return outputs, enhanced_image, A

# ==========================================
# 2. RUN THE BENCHMARK
# ==========================================
def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_path = "../epoch_files/joint_model_epoch_60.pt"
    
    # Load your trained weights
    model = JointModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # DETR standard input size is usually 800x800 after processing
    dummy_input = torch.randn(1, 3, 800, 800).to(device)

    print("🔥 Warming up GPU...")
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    print("⏱️ Starting benchmark over 100 iterations...")
    times = []
    with torch.no_grad():
        for i in range(100):
            start = time.time()
            _ = model(dummy_input)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            end = time.time()
            times.append((end - start) * 1000)

    avg_time = np.mean(times)
    fps = 1000 / avg_time
    print(f"\n--- 📊 Final Performance Metrics ---")
    print(f"Average Inference Time: {avg_time:.2f} ms")
    print(f"Throughput (FPS):      {fps:.2f} frames/sec")
    print(f"------------------------------------")

if __name__ == "__main__":
    run_benchmark()


# .\venv\Scripts\activate
# cd main_python_files
# python benchmark.py
