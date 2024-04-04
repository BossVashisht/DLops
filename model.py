import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_curve
import numpy as np

# Load USPS dataset
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

train_dataset = datasets.USPS(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.USPS(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 64 * 5 * 5)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the MLP architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())
    return predictions, true_labels

# Initialize the CNN model, loss function, and optimizer
cnn_model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Train the CNN model
train_model(cnn_model, train_loader, criterion, optimizer)

# Evaluate the CNN model
cnn_predictions, true_labels = evaluate_model(cnn_model, test_loader)

# Compute metrics for CNN
cnn_accuracy = accuracy_score(true_labels, cnn_predictions)
cnn_precision = precision_score(true_labels, cnn_predictions, average='macro')
cnn_recall = recall_score(true_labels, cnn_predictions, average='macro')
cnn_conf_matrix = confusion_matrix(true_labels, cnn_predictions)

# Initialize the MLP model, loss function, and optimizer
mlp_model = MLP()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

# Train the MLP model
train_model(mlp_model, train_loader, criterion, optimizer)

# Evaluate the MLP model
mlp_predictions, true_labels = evaluate_model(mlp_model, test_loader)

# Compute metrics for MLP
mlp_accuracy = accuracy_score(true_labels, mlp_predictions)
mlp_precision = precision_score(true_labels, mlp_predictions, average='macro')
mlp_recall = recall_score(true_labels, mlp_predictions, average='macro')
mlp_conf_matrix = confusion_matrix(true_labels, mlp_predictions)

# Compare the models
print("CNN Metrics:")
print(f"Accuracy: {cnn_accuracy}")
print(f"Precision: {cnn_precision}")
print(f"Recall: {cnn_recall}")
print("Confusion Matrix:")
print(cnn_conf_matrix)

print("\nMLP Metrics:")
print(f"Accuracy: {mlp_accuracy}")
print(f"Precision: {mlp_precision}")
print(f"Recall: {mlp_recall}")
print("Confusion Matrix:")
print(mlp_conf_matrix)

# Write metrics to TensorBoard
writer = SummaryWriter()

# CNN metrics
writer.add_scalar('CNN/Accuracy', cnn_accuracy)
writer.add_scalar('CNN/Precision', cnn_precision)
writer.add_scalar('CNN/Recall', cnn_recall)

# MLP metrics
writer.add_scalar('MLP/Accuracy', mlp_accuracy)
writer.add_scalar('MLP/Precision', mlp_precision)
writer.add_scalar('MLP/Recall', mlp_recall)


writer.close()
