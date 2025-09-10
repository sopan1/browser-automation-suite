import torch
import torch.nn as nn
import numpy as np
import time

class MouseLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=2, output_size=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


def simulate_mouse_movements(num_points=20):
    model = MouseLSTM()
    x = np.zeros((1, num_points, 2))
    for i in range(1, num_points):
        x[0, i, 0] = x[0, i-1, 0] + np.random.randint(-10, 10)
        x[0, i, 1] = x[0, i-1, 1] + np.random.randint(-10, 10)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    out = model(x_tensor).detach().numpy()
    path = []
    for i in range(num_points):
        path.append({
            "x": int(out[0][i][0]),
            "y": int(out[0][i][1]),
            "t": time.time() + i * 0.05
        })
    return path

if __name__ == "__main__":
    import json
    print(json.dumps(simulate_mouse_movements()))