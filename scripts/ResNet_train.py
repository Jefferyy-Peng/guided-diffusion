import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset, random_split
import random
import os
import pickle

# Set device: use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for the dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to 256x256
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize to ImageNet stats
    ])
}

# Set up the dataset (assuming you have a 'train' folder with 100 images)
data_dir = '../datasets/imagenet1k/ILSVRC/Data/CLS-LOC'  # Path to your dataset


image_datasets = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])

# Select the target classes you are interested in (97, 130, 439, 863)
selected_class_indices = [0, 1, 2, 3]
class_to_idx = image_datasets.class_to_idx

# dataloaders = {x: DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
#                for x in ['train']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
# class_names = image_datasets['train'].classes

# Get indices of images that belong to the selected classes
# selected_indices = []
# for target_class in selected_class_indices:
#     count = 0
#     for i, (img, label) in enumerate(image_datasets):
#         if label == target_class:
#             selected_indices += [i]
#             count += 1
#         if count > 50:
#             break
# with open('../data_index.p', 'wb') as f:
#     pickle.dump(selected_indices, f)

with open('../data_index.p', 'rb') as f:
    selected_indices = pickle.load(f)
# Create a subset with only the selected images
subset_dataset = Subset(image_datasets, selected_indices)

train_size = int(0.8 * len(subset_dataset))  # 80% training, 20% validation
val_size = len(subset_dataset) - train_size
train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])

# DataLoader for the selected subset
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

# Get size of subset dataset
subset_size = len(subset_dataset)

# Get class names
subset_class_names = [image_datasets.classes[idx] for idx in selected_class_indices]


# Load the pre-trained ResNet model and modify for our dataset
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(subset_class_names))  # Adjust output layer for the number of classes

# Move the model to the GPU if available
model = model.to(device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
best_val_acc = 0.0  # Track the best validation accuracy
num_epochs = 100
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 10)

    # Train the model
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / train_size
    epoch_acc = running_corrects.double() / train_size

    print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    # Evaluate on validation set
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss = val_loss / val_size
    val_acc = val_corrects.double() / val_size

    print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    # Save model if it has the best validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        print(f'Saving model with validation accuracy: {val_acc:.4f}')
        torch.save(model.state_dict(), '../models/best_resnet_model.pth')

