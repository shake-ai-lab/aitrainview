"""Any PyTorch model, any training loop: a small CNN trained on synthetic images, watched in AITrainView.

Start the app first, then:  python examples/custom_loop_cnn.py
The AITrainView-specific lines are `watch(model)`, `w.step(loss)` and `w.close()`.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))  # without pip install -e
from aitrainview.watch import watch  # noqa: E402


class SmallCNN(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def main(steps: int = 200):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = SmallCNN().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # synthetic task: the class is encoded in the colour / position of a bright square
    def batch(n=64):
        y = torch.randint(0, 10, (n,), device=dev)
        x = torch.randn(n, 3, 32, 32, device=dev) * 0.3
        for i in range(n):
            c, k = int(y[i]) % 3, int(y[i])
            x[i, c, (k // 4) * 8:(k // 4) * 8 + 8, (k % 4) * 8:(k % 4) * 8 + 8] += 2.0
        return x, y

    w = watch(model, steps=steps, name="SmallCNN (custom loop)")  # ← AITrainView
    for step in range(1, steps + 1):
        x, y = batch()
        loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        w.step(loss)  # ← after backward(), before optimizer.step()
        opt.step()
        opt.zero_grad()
        if w.stop_requested:  # the stop button in the app
            break
        if step % 50 == 0:
            print(f"step {step}: loss {loss.item():.3f}")
    w.close()  # ← AITrainView


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
