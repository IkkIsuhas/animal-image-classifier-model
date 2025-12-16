# importing all the dependencies
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("using device: ",device)

data = "./dataset"
test = "./bear.jpg"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

dataset = datasets.ImageFolder(root=data,transform=transform)

num_class = len(dataset.classes)
class_names = dataset.classes

print("Classes: ",class_names)

train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_class)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5

print("Starting Training process let's gooo!!!")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()

    print(f"Epoch {epoch+1}/{epochs} - loss: {running_loss}")

print("Training Completed!!!")

torch.save(model.state_dict(), "animal_resnet.pth")
print("Model saved")

model.load_state_dict(
    torch.load("animal_resnet.pth", map_location=device)
)
model.eval()
image = Image.open(test).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image_tensor)

probabilities = torch.softmax(output, dim=1)
predicted_index = probabilities.argmax(dim=1).item()

predicted_class = class_names[predicted_index]
confidence = probabilities[0][predicted_index].item() * 100

print("\nPrediction Result")
print("Image:", test)
print("Predicted class:", predicted_class)
print(f"Confidence: {confidence:.2f}%")