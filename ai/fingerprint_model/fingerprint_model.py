import torch
import torch.nn as nn
import random

class FingerprintModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

def generate_fingerprint():
    # Simulate model input/output for demonstration
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/115.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) Firefox/102.0"
    ]
    screen_sizes = [(1920,1080), (1366,768), (1600,900)]
    colors = [24, 32]
    sample = torch.randn(10)
    model = FingerprintModel()
    output = model(sample)
    fingerprint = {
        "userAgent": random.choice(user_agents),
        "screen": {
            "width": random.choice(screen_sizes)[0],
            "height": random.choice(screen_sizes)[1],
            "colorDepth": random.choice(colors)
        },
        "webgl_vendor": "NVIDIA Corporation",
        "canvas_hash": hex(random.getrandbits(128)),
        "audio_hash": hex(random.getrandbits(64))
    }
    return fingerprint

if __name__ == "__main__":
    import json
    print(json.dumps(generate_fingerprint()))