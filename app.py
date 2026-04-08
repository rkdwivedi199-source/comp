from fastapi import FastAPI, File, UploadFile
import torch
import pickle
import io
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

app = FastAPI(title="MNIST CNN API")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CNN Model (same as training)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Load model
model = CNN().to(device)

with open("cnn_mnist_weights.pkl", "rb") as f:
    model.load_state_dict(pickle.load(f))

model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.get("/")
def home():
    return {"message": "MNIST CNN API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("L")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return {
        "filename": file.filename,
        "predicted_digit": int(predicted.item())
    }
