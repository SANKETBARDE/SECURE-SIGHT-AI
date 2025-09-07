import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

# CNN + LSTM model definition (same as training)
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

# Label mapping for binary classification
label_mapping = {0: "Normal", 1: "Crime"}

# Video and model paths - change as needed
video_path = "C:/Users/Lenovo/OneDrive/Desktop/FINAL PROJECT/normal.mp4"
model_path = "C:/Users/Lenovo/OneDrive/Desktop/FINAL PROJECT/best_model.pth"

# Transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load and preprocess video frames
def load_video_frames(path, num_frames=16):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = transform(img)
        frames.append(img)
    cap.release()

    # Handle insufficient frames
    if len(frames) < num_frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    else:
        # Select center 16 frames
        center = len(frames) // 2
        start = max(0, center - num_frames // 2)
        frames = frames[start:start + num_frames]

    return torch.stack(frames).unsqueeze(0)  # Shape: [1, 16, 3, 64, 64]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Predict with probability
with torch.no_grad():
    input_clip = load_video_frames(video_path).to(device)
    output = model(input_clip)  # raw logits
    probs = F.softmax(output, dim=1)  # convert logits to probabilities
    pred = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred].item() * 100

    print(f"\n🧠 Prediction: {label_mapping[pred]} ({confidence:.2f}% confidence)")
    print(f"📊 Class probabilities → Normal: {probs[0][0].item():.4f}, Crime: {probs[0][1].item():.4f}")
