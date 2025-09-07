import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ✅ Class mapping
label_mapping = {
    0: 'Assault',
    1: 'Robbery'
}

# ✅ Define the model (same as training)
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)
        features = features.view(B, T, 128)
        lstm_out, _ = self.lstm(features)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out

# ✅ Load model
model = CNNLSTM(num_classes=2)
model.load_state_dict(torch.load("best_model.pth", map_location='cpu'))
model.eval()

# ✅ Load 16 test frames
test_dir = "test_clip"  # Folder with 16 PNG frames
transform = transforms.Compose([
    transforms.ToTensor()
])

frames = []
frame_files = sorted(os.listdir(test_dir))[:16]  # Get first 16 frames
for file in frame_files:
    img_path = os.path.join(test_dir, file)
    img = Image.open(img_path).convert("RGB")
    img = transform(img)
    frames.append(img)

# ✅ Stack & run through model
input_clip = torch.stack(frames).unsqueeze(0)  # Shape: [1, 16, 3, 64, 64]
with torch.no_grad():
    output = model(input_clip)
    pred = torch.argmax(output, dim=1).item()

print(f"🧠 Predicted Class: {label_mapping[pred]}")
