import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


# ✅ Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Image preprocessing (must match training preprocessing)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ✅ Load dataset just to get class names (not for training now)
train_data = datasets.ImageFolder("data/train", transform=transform)

# ✅ Define model architecture (must be same as training)
class FoodCNN(nn.Module):
    def __init__(self):
        super(FoodCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, len(train_data.classes))
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# ✅ Create model instance
model = FoodCNN().to(device)

# ✅ Load saved weights
model.load_state_dict(torch.load("nigeria_food_cnn.pth", map_location=device))
model.eval()
print("✅ Model loaded successfully!")

# ✅ Prediction function
def predict_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    output = model(img)
    pred = torch.argmax(output, dim=1).item()
    return train_data.classes[pred]

# ✅ Test with multiple images
test_images = [
    "text_images/galas.jfif",
    "text_images/imagine.jfif",
    "text_images/imoo.jpg",
    "text_images/sharp.jpg",
    "text_images/towhere.jpg",
    "text_images/unknown.jpg"
]

for img_path in test_images:
    try:
        prediction = predict_image(img_path)
        print(f"{img_path}: {prediction}")
    except Exception as e:
        print(f"Error with {img_path}: {e}")



      

 


