import torch
import time
import numpy as np
from models import JointModel

class Benchmarker:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = JointModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def run_benchmark(self, iterations=100):
        dummy_input = torch.randn(1, 3, 800, 800).to(self.device)
        print("🔥 Warming up...")
        for _ in range(10): 
            with torch.no_grad(): _ = self.model(dummy_input)

        print(f"⏱️ Running {iterations} iterations...")
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.time()
                _ = self.model(dummy_input)
                if self.device.type == 'cuda': torch.cuda.synchronize()
                times.append((time.time() - start) * 1000)

        print(f"📊 Avg Inference: {np.mean(times):.2f} ms | FPS: {1000/np.mean(times):.2f}")

if __name__ == "__main__":
    tester = Benchmarker("../epoch_files/joint_model_epoch_60.pt")
    tester.run_benchmark()