import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    'Bear', 'Bird', 'Cat', 'Cow', 'Deer',
    'Dog', 'Dolphin', 'Elephant', 'Giraffe',
    'Horse', 'Kangaroo', 'Lion', 'Panda',
    'Tiger', 'Zebra'
]

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("animal_resnet.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

image = Image.open("dalphin.jpg").convert("RGB")
image = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)

pred = output.argmax(dim=1).item()
probabilities = torch.softmax(output, dim=1)
predicted_index = probabilities.argmax(dim=1).item()
confidence = probabilities[0][predicted_index].item() * 100
print("Predicted class:", class_names[pred])
print(f"Confidence: {confidence:.2f}%")