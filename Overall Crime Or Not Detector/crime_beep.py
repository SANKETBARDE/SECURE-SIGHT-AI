import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Define your CNNLSTM model (must match your training architecture)
class CNNLSTM(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(CNNLSTM, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.lstm = torch.nn.LSTM(input_size=128, hidden_size=256, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        features = self.cnn(x)
        features = features.view(B, T, 128)
        lstm_out, _ = self.lstm(features)
        last_out = lstm_out[:, -1, :]
        out = self.fc(last_out)
        return out

# Paths to your video file and model weights - change these!
video_path = "C:/Users/Lenovo/OneDrive/Desktop/FINAL PROJECT/Man Who Killed Victim With Single Punch Jailed CCTV.mp4"
model_path = "C:/Users/Lenovo/OneDrive/Desktop/FINAL PROJECT/best_model.pth"

# Load model and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNLSTM(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("FPS not detected, defaulting to 25.")
    fps = 25
frame_duration_ms = int(1000 / fps)

num_frames_per_clip = 16
frame_buffer = []

crime_detected = False
crime_start_time = None
label_mapping = {0: "Normal", 1: "Crime"}

print(f"Processing video: {video_path}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached.")
        break

    # Preprocess frame for model input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    tensor_img = transform(pil_img)
    frame_buffer.append(tensor_img)

    # Once enough frames are buffered for a clip, run inference
    if len(frame_buffer) == num_frames_per_clip:
        clip_tensor = torch.stack(frame_buffer).unsqueeze(0).to(device)  # Shape: [1, T, C, H, W]

        with torch.no_grad():
            output = model(clip_tensor)
            probs = F.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()

        # Calculate clip start time in seconds
        current_frame_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        clip_start_time = max(0, (current_frame_pos - num_frames_per_clip) / fps)

        # Detect crime start timestamp
        if pred == 1 and not crime_detected:
            crime_detected = True
            crime_start_time = clip_start_time
            print(f"[ALERT] Crime started at {crime_start_time:.2f} seconds.")

        elif pred == 0 and crime_detected:
            crime_detected = False  # Crime ended, you can also log this

        # Display status on video window
        status_text = f"Status: {label_mapping[pred]}"
        if crime_detected:
            status_text += f" (Started at {crime_start_time:.2f}s)"
        color = (0, 0, 255) if pred == 1 else (0, 255, 0)  # Red for crime, green for normal

        cv2.putText(frame, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Remove oldest frame to slide the clip window
        frame_buffer.pop(0)

    # Show video frame with overlay
    cv2.imshow("Crime Detection", frame)

    # Press 'q' to quit early
    if cv2.waitKey(frame_duration_ms) & 0xFF == ord('q'):
        print("Interrupted by user.")
        break

cap.release()
cv2.destroyAllWindows()
