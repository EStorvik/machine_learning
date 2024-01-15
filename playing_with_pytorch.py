# Step 1: Set Up the Environment
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# Step 2: Create a Neural Network Class
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 64)  # First fully connected layer
        self.fc2 = nn.Linear(64, 10)     # Output layer (10 classes for MNIST)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1) # Flatten the image
        x = torch.relu(self.fc1(x))       # Activation function after first layer
        x = self.fc2(x)                   # Output layer, no activation (will use CrossEntropyLoss)
        return x

# Step 3: Load and Preprocess Data
transform = transforms.ToTensor()
train_data = MNIST(root='./data', train=True, download=True, transform=transform)
test_data = MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# Step 4: Instantiate the Network, Define Loss Function and Optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()  # Loss function for classification
# optimizer = optim.SGD(model.parameters(), lr=0.001)  # Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimizer

# Step 5: Train the Network
for epoch in range(5):  # 5 epochs
    for data in train_loader:
        inputs, labels = data
        optimizer.zero_grad()   # Zero the gradient buffers
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        loss.backward()         # Backward pass
        optimizer.step()        # Update weights

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Step 6: Evaluate the Network
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on test images: {100 * correct // total} %')
