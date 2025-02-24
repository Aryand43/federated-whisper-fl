import flwr as fl  # Flower framework for Federated Learning
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # First layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # Output layer

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Load dataset (MNIST as an example)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define training function
def train(model, train_loader, epochs=1):
    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer
    
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

# Federated Learning Client Class
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model

    def get_parameters(self) -> List:
        """Extract model weights to send to server."""
        return [param.detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters: List) -> None:
        """Update model weights with received parameters."""
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param)

    def fit(self, parameters, config) -> tuple:
        """Train model locally and send updated weights to server."""
        self.set_parameters(parameters)  # Set initial model weights
        train(self.model, train_loader, epochs=1)  # Train model
        return self.get_parameters(), len(train_loader.dataset), {}

    def evaluate(self, parameters, config) -> tuple:
        """Evaluate model performance on test set."""
        self.set_parameters(parameters)
        self.model.eval()

        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return float(0), len(test_loader.dataset), {"accuracy": accuracy}  # Dummy loss value

# Start the client and connect to the server
model = SimpleNN()
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(model))
