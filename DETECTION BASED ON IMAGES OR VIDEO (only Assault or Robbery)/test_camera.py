import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# Define CNN+LSTM model with the correct dimensions based on the checkpoint
class CNNLSTMModel(torch.nn.Module):
    def __init__(self):
        super(CNNLSTMModel, self).__init__()

        # CNN layers with correct number of filters
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 32 filters
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 filters
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # LSTM with corrected dimensions based on the original model
        self.lstm = torch.nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True)

        # Fully connected layer with the correct dimensions (based on the saved model)
        self.fc = torch.nn.Linear(256, 2)  # Output for 2 classes: robbery or assault

    def forward(self, x):
        x = self.cnn(x)  # Pass through CNN layers
        
        # Calculate the flattened size: number of features after CNN
        cnn_out_size = x.size(1) * x.size(2) * x.size(3)  # Channels * Height * Width
        
        # Flatten the CNN output for LSTM
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, cnn_out_size)
        
        # Ensure the input to LSTM is of the correct size
        x = x.view(x.size(0), 1, cnn_out_size)  # (batch_size, seq_len=1, input_size=cnn_out_size)
        
        # Now the input is correctly shaped for LSTM
        x, _ = self.lstm(x)  # Pass through LSTM
        
        # Use the last LSTM output for classification
        x = self.fc(x[:, -1, :])  # Use the last time-step output (batch_size, hidden_size)
        return x

# Create the model with the correct dimensions
model = CNNLSTMModel()

# Load the trained model weights
model.load_state_dict(torch.load('C:/Users/Lenovo/OneDrive/Desktop/crime/best_model.pth'), strict=False)

# Set model to evaluation mode
model.eval()

# Define the class labels for prediction
class_labels = ['robbery', 'assault']

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224 for the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize based on the model's training
])

# Function to preprocess input data (single image or video frame)
def preprocess_input(input_data):
    input_data = transform(input_data)
    input_data = input_data.unsqueeze(0)  # Add batch dimension
    return input_data

# Function to make predictions
def predict_action(input_data):
    processed_input = preprocess_input(input_data)
    with torch.no_grad():
        output = model(processed_input)
    _, predicted_class_index = torch.max(output, 1)
    predicted_class = class_labels[predicted_class_index.item()]
    return predicted_class

# Example: Load and classify a PNG image
image_path = 'C:/Users/Lenovo/OneDrive/Desktop/crime/Robbery048_x264_0.png'
input_image = Image.open(image_path).convert('RGB')  # Open the PNG and convert to RGB
predicted_class = predict_action(input_image)
print(f"Predicted Action: {predicted_class}")

# For video stream (webcam or video file)
cap = cv2.VideoCapture(0)  # Use 0 for default webcam or provide a video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    predicted_class = predict_action(frame)
    
    # Display the frame with the predicted action
    cv2.putText(frame, f"Action: {predicted_class}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Video Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
