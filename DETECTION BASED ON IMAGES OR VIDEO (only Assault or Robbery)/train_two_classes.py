import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# ✅ Dataset class
def get_label_mapping():
    return {
        'Assault': 0,
        'Robbery': 1
    }

class CrimeFramesDataset64(Dataset):
    def __init__(self, root_dir, label_mapping, frames_per_clip=16):
        self.root_dir = root_dir
        self.label_mapping = label_mapping
        self.frames_per_clip = frames_per_clip
        self.data = []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        for class_name in os.listdir(root_dir):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue

            frame_files = sorted(os.listdir(class_folder))
            for i in range(0, len(frame_files) - frames_per_clip, frames_per_clip):
                clip = frame_files[i:i+frames_per_clip]
                self.data.append((class_folder, clip, label_mapping[class_name]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        class_folder, clip_filenames, label = self.data[idx]
        frames = []

        for filename in clip_filenames:
            frame_path = os.path.join(class_folder, filename)
            img = Image.open(frame_path).convert('RGB')
            img = self.transform(img)
            frames.append(img)

        frames = torch.stack(frames)  # [16, 3, 64, 64]
        return frames, label

# ✅ CNN + LSTM model
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

# ✅ Configurations
root_path = "C:/Users/Lenovo/OneDrive/Desktop/crime/dataset"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 5
batch_size = 4

label_mapping = get_label_mapping()
dataset = CrimeFramesDataset64(root_dir=root_path, label_mapping=label_mapping, frames_per_clip=16)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

model = CNNLSTM(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ✅ Training + Save + Plot
train_losses = []
train_accuracies = []
best_acc = 0

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total = 0

    print(f"\n🔁 Epoch {epoch + 1}/{epochs}")
    pbar = tqdm(train_loader, desc="Training", unit="batch")

    for frames, labels in pbar:
        frames, labels = frames.to(device), labels.to(device)

        outputs = model(frames)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        acc = correct / total * 100
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{acc:.2f}%"})

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    train_accuracies.append(acc)

    print(f"✅ Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} | Final Acc: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"📦 Best model saved (acc = {acc:.2f}%)")

# ✅ Plot loss and accuracy
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o', label="Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, marker='o', color='green', label="Accuracy")
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True)

plt.tight_layout()
plt.show()
