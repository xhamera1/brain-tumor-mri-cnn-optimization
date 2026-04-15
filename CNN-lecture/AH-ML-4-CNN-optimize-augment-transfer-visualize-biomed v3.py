# %% [markdown]
# <table>
# <tr>    
# <td style="text-align: center">
# <h1>Introduction to Comprehensive Training, Augmentation, Transfer Learning and Visualization of CNNs on Biomedical Data</h1>
# <h2><a href="http://home.agh.edu.pl/~horzyk/index.php">Adrian Horzyk</a></h2>
# </td> 
# <td>
# <img src="http://home.agh.edu.pl/~horzyk/im/AdrianHorzyk49BT140h.png" alt="Adrian Horzyk, Professor" title="Adrian Horzyk, Professor" />        
# </td> 
# </tr>
# </table>
# <h3><i>Welcome to the interactive lecture and exercises where you can check everything and experiment!</i></h3>

# %% [markdown]
# 
# # Comprehensive CNN Training, Augmentation, Transfer Learning & Visualization using Biomedical Data
# 
# This notebook demonstrates:
# - **CNN Training:** Building and training a convolutional network from scratch.
# - **Data Augmentation:** Improving model generalization using augmentation techniques.
# - **Transfer Learning:** Leveraging a pre-trained VGG16 model to fine-tune on your dataset.
# - **Visualization:** Analyzing training performance and inspecting intermediate CNN representations and filters.
# 
# **Table of Contents:**
# 1. Setup and Imports
# 2. Data Preprocessing & Augmentation
# 3. Building a simple CNN from Scratch
# 4. Transfer Learning with **VGG16**, **VGG19**, **ResNet50**, **InceptionV3**, **MobileNetV2**, **EfficientNetB0**
# 5. Visualization Techniques **VGG16**, **VGG19**, **ResNet50**, **InceptionV3**, **MobileNetV2**, **EfficientNetB0**
# 6. Conclusion & Further Work

# %% [markdown]
# ## Deep learning for small training data problems
# 
# Maybe you heard that deep learning only works when lots of data are available. This is, in part, a valid point because one fundamental characteristic of deep learning is that many training data are able to find interesting and enough frequent features in these data without manual feature engineering. This can only be achieved when lots of training examples are available. This is especially true for problems where the input samples are very high-dimensional, like images, where the input space dimension is defined by the image resolution.
# 
# However, what constitutes "lots" of samples is relative to the size and depth of the network you are trying to train. It isn't possible to train a convnet to solve a complex problem with just a few tens of examples, but a few hundred can sometimes suffice if the model is small and well-regularized and if the task is simple. 
# 
# Because convnets learn local, translation-invariant features, they are very data-efficient on perceptual problems. Training a convnet from scratch on a very small image dataset will still yield reasonable results despite a relative lack of data, without the need for any custom feature engineering. You will try to do it here in this notebook!
# 
# What's more, deep learning models are, by nature, highly repurposable. It means that you can take an image classification or speech-to-text model trained on a large-scale dataset and then reuse it on a significantly different problem with only minor changes. This we call <b>transfer learning</b>. Specifically, in the case of computer vision, many pre-trained models (usually trained on the ImageNet dataset) are publicly available for download and can be used to bootstrap powerful vision models out of very little data.
# 
# Having to train an image classification model using only very little data is a common situation that you likely encounter yourself in practice in a professional context.
# 
# Having "few" samples can mean anywhere from a few hundred to few tens of thousands of images. The definition of "few" depends on how many data dimensions we have. We demonstrate it by classifying images of MedMNIST dataset collection. Our workflow will be as follows:
# 
# * First, we will upload this dataset that is already divided into training, validation, and testing images.
# 
# * Next, we will train a new model from scratch, starting with a small convnet, without any regularization, to set a baseline for what can be achieved later.
# 
# * Then, we will introduce <b>data augmentation</b>, a powerful technique for mitigating overfitting in computer vision. By leveraging data augmentation, we will improve our network to reach a higher accuracy of the model if we set up the augmentation hyperparameters correctly.
# 
# * Finally, we will apply <b>feature extraction with a pre-trained network</b>, and <b>fine-tuning a pre-trained network</b> to increase the final model peformance.
# 
# These strategies together will constitute your basic future toolbox for tackling the problem of doing computer vision with small datasets. Next, we use <b>transfer learning</b> to train the model faster and better, using pre-trained models on a big dataset to use well-trained features to adapt the model for similar tasks easier and faster!
# 
# For now, let's get started by getting our hands on the data and preparing them for our experiments.

# %%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# %% [markdown]
# We will use the “PathMNIST” dataset from the MedMNIST collection. PathMNIST is a collection of histopathologic images, and MedMNIST conveniently provides predefined train/validation/test splits, so you won’t have to perform manual splitting.

# %% [markdown]
# ## 1 Install and Import the Required Package and Download Data
# 
# Add a cell near the top of your notebook (if not already present) to install and import the MedMNIST package along with other required libraries. For example:

# %%
# Install medmnist package if not already installed
!pip install medmnist

!pip uninstall typing_extensions -y
!pip install typing_extensions==4.10.0
!pip install --upgrade torch torchvision

# Restart your runtime (or Python interpreter) to ensure that the new package version is loaded.

# %% [markdown]
# Use the following code to download and prepare the PathMNIST dataset. In this example, we assume the images are RGB; adjust the normalization if needed.

# %%
# Import libraries
import medmnist
from medmnist import INFO  # Provides metadata about the dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# %%
# You could need to update:
!pip install --upgrade typing_extensions

# %%
# Specify which MedMNIST dataset to use (here: 'pathmnist' for histopathologic images)
data_flag = 'pathmnist'
download = True  # This will download the dataset if it's not already available locally
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])
# DataClass = getattr(medmnist.dataset, info['python_class'])   # use this if not working due to on another version of MedMNIST

# Define normalization parameters (assumed RGB images; modify if necessary)
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# Create transforms including any augmentation for training data
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),  # Data augmentation example
    transforms.Normalize(mean, std)
])

# For validation and test, use only basic transforms
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load datasets using predefined splits provided by MedMNIST
train_dataset = DataClass(split='train', transform=train_transform, download=download)
val_dataset = DataClass(split='val', transform=val_transform, download=download)
test_dataset = DataClass(split='test', transform=test_transform, download=download)

# ----------------------------------
# Option: Reduce Dataset Sizes to accelerate training
# ----------------------------------
reduce_dataset = True  # Set to True to reduce dataset sizes for faster training
if reduce_dataset:
    from torch.utils.data import Subset
    import numpy as np
    print("Reducing dataset sizes for faster training.")
    reducedivider = 8   # Reduce the dataset if necesary or increase it to get better results
    # Generate a random permutation of indices for each dataset
    train_indices = np.random.permutation(len(train_dataset))[:len(train_dataset) // reducedivider]
    val_indices = np.random.permutation(len(val_dataset))[:len(val_dataset) // reducedivider]
    test_indices = np.random.permutation(len(test_dataset))[:len(test_dataset) // reducedivider]
    # Create new subsets using the randomized indices
    train_dataset = Subset(train_dataset, train_indices)
    val_dataset = Subset(val_dataset, val_indices)
    test_dataset = Subset(test_dataset, test_indices)

# Create dataloaders for each split
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Dataset loaded: {} training images, {} validation images, {} test images".format(
    len(train_dataset), len(val_dataset), len(test_dataset)
))


# %% [markdown]
# ## 2. Preparing Convolutional Neural Network for Biomedical Images
# 
# There is a Python code snippet that defines, initializes, and constructs a convolutional neural network (CNN) architecture for biomedical (RGB) images. This example uses two convolutional layers, pooling, and fully connected layers. You can integrate this code into your notebook as the cell where you define your model.
# 
# 
# ### Imports
# - We import the necessary modules from PyTorch.
# 
# ### CNN Class
# - **Constructor (`__init__`)**:  
#   Defines two convolution layers (`conv1` and `conv2`) that transform the input (with three channels for RGB images) into 32 and then 64 feature maps.
# - A max pooling layer (`pool`) reduces the spatial dimensions by half each time.
# - The fully connected layers (`fc1` and `fc2`) transform the flattened feature maps into the final output (with a number of neurons equal to `num_classes`).
# - A dropout layer is added after the first fully connected layer for regularization.
# 
# ### Forward Pass
# - The `forward` method applies the convolutional layers with ReLU activations, pooling, flattens the output, then passes it through the fully connected layers.
# 
# ### Model Initialization
# - We create an instance of the CNN with the appropriate number of classes, input channels, and image size.
# - Finally, we move the model to GPU if available and print the model summary.
# 
# This complete code snippet should serve as the foundation for your biomedical image analysis notebook using convolutional neural networks.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self, num_classes=9, input_channels=3, image_size=28):
        """
        Args:
            num_classes (int): Number of classes in the dataset.
            input_channels (int): Number of channels in the input image. For biomedical RGB images, set to 3.
            image_size (int): Height/width of the input images (assumes square images).
        """
        super(CNN, self).__init__()
        # First convolution layer: from 3 channels (RGB) to 32 feature maps
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        # Second convolution layer: from 32 to 64 feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Pooling layer to reduce spatial dimensions by a factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the feature map size after two pooling layers
        # (image_size -> image_size/2 -> image_size/4)
        fc_input_dim = 64 * (image_size // 4) * (image_size // 4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        # Optional dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Pass through first convolutional layer, activation, and pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # Pass through second convolutional layer, activation, and pooling
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        # First fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # Final output layer (no activation; e.g., logits for classification)
        x = self.fc2(x)
        return x

# Example of initializing the model
# Update num_classes to match your dataset. For instance, if using MedMNIST 'pathmnist', use info['n_classes'].
num_classes = 9  # Replace with actual number of classes if different.
model = CNN(num_classes=num_classes, input_channels=3, image_size=28)

# Optionally, move the model to GPU if available:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Print the model architecture
print(model)


# %% [markdown]
# ## 3. Training, Optimization, and Evaluation
# 
# Since your notebook already includes cells for model training, optimization, augmentation demonstrations, transfer learning, and visualization, you can keep those sections largely the same. They will now operate on the biomedical dataset provided by PathMNIST. The following code snippet shows how you can initialize the model, define the loss and optimizer, and perform training as well as validation.
# 
# Below is an explanation of the key steps:
# 
# - **Model Initialization:**  
#   We create an instance of the CNN model (designed for RGB images) with the correct number of classes (as provided by the dataset). We then move the model to the GPU if available.
# 
# - **Loss Function & Optimizer:**  
#   We use `CrossEntropyLoss` for multi-class classification and the Adam optimizer for updating model weights.
# 
# - **Training Loop:**  
#   For each epoch, the model is set to training mode and iterates over the training dataset. The gradients are zeroed, the forward pass is computed, the loss is calculated, and backpropagation is performed before updating the weights.
# 
# - **Validation:**  
#   After each epoch, the model is set to evaluation mode. We compute the loss and accuracy over the validation dataset, which helps in tracking performance during training.
# 
# This section forms the basis of training your convolutional neural network on the biomedical images.

# %%
import torch
import torch.nn as nn
import torch.optim as optim

# Assume the CNN model is already defined as shown in previous sections
# and the train_loader, val_loader (and test_loader if needed) are set up from the PathMNIST dataset.

# Example: set number of classes using dataset information, e.g., for PathMNIST
num_classes = 9  # Replace with info['n_classes'] if available
image_size = 28  # Adjust this based on the dataset; PathMNIST images are 28x28

# Initialize the model (ensure the CNN model accepts 3-channel RGB images)
model = CNN(num_classes=num_classes, input_channels=3, image_size=image_size)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs for training
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Training loop
    for inputs, labels in train_loader:
        # Move inputs and labels to the device
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.squeeze()  # Ensure labels have shape [batch_size]
        
        optimizer.zero_grad()           # Clear the gradients
        outputs = model(inputs)         # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()                 # Backpropagation
        optimizer.step()                # Update weights
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()  # Ensure labels have shape [batch_size]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Get predictions and calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

# For testing, you would use a similar loop with test_loader to evaluate final performance.


# %% [markdown]
# ### Summary
# 
# - **Dataset Change:**  
#   The original dataset loading is replaced with code that downloads and splits the biomedical PathMNIST dataset from MedMNIST.
# 
# - **Data Augmentation & Normalization:**  
#   A transforms pipeline is provided that includes data augmentation (e.g., random horizontal flipping) for training and normalization for training, validation, and test splits.
# 
# - **CNN Input Adjustment:**  
#   The CNN architecture is adjusted to accept 3-channel (RGB) images by setting the first convolution layer’s input channels to 3.
# 
# - **Dataloaders:**  
#   Separate data loaders are set up for training, validation, and test sets using the splits provided by MedMNIST.
# 

# %% [markdown]
# ## 4. Data Augmentation
# 
# 
# ### Augmentation Options Preview
# 
# We define a dictionary (`augmentation_options`) with several common augmentation transforms such as random horizontal/vertical flips, random rotations, color jitter, random affine transformations, and random resized crop.
# 
# We load a sample image from the training dataset (converted to a PIL image) and apply each augmentation. This allows you to see visually how each augmentation alters the image.
# 
# ### Chosen Augmentation Pipeline
# 
# For training, we compose a new transform (`chosen_train_transform`) that includes a subset of augmentations (horizontal flip, a small rotation, color jitter) followed by normalization.
# 
# The validation and test transforms remain simple (only conversion to tensor and normalization) since we generally do not augment these sets.
# 
# ### Loading Datasets and DataLoaders
# 
# The training, validation, and test datasets are loaded with the appropriate transforms, and DataLoaders are created to feed the data into the model.
# 
# This setup not only makes the augmentation steps visible for educational purposes but also allows you to easily experiment with different augmentation functions and their parameters. You can modify the `chosen_train_transform` or any of the individual augmentations in the preview section to better understand their effect and potentially improve performance.
# 

# %%
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt
from PIL import Image

# %%
# -----------------------
# Dataset Loading & Preprocessing
# -----------------------

# Specify the MedMNIST dataset to use (PathMNIST for biomedical images)
data_flag = 'pathmnist'
download = True  # Set to True to download the dataset if not available locally
info = INFO[data_flag]
print("INFO keys:", list(info.keys()))  # Debug: print keys available in info

DataClass = getattr(medmnist, info['python_class'])

# Try to get number of classes from the info dictionary.
num_classes = info.get('n_classes', 9)  # Default to 9 for PathMNIST if missing

# Define normalization parameters (assumes images are RGB)
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# %% [markdown]
# In the following code, there are several common data augmentation functions. In this version, we create an “augmentation preview” section so you can see how each augmentation affects a sample image. You can then modify and combine these augmentations to see their impact on the training process.

# %%
# ----------------------------------------------
# Augmentation Options (For Educational Purposes)
# ----------------------------------------------

# We'll define several augmentations so you can see and experiment with them.
# Here are a few examples:
augmentation_options = {
    "RandomHorizontalFlip": transforms.RandomHorizontalFlip(p=1.0),  # Always flip horizontally
    "RandomVerticalFlip": transforms.RandomVerticalFlip(p=1.0),      # Always flip vertically
    "RandomRotation_30": transforms.RandomRotation(30),               # Rotate by up to ±30 degrees
    "ColorJitter": transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    "RandomAffine": transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    "RandomResizedCrop": transforms.RandomResizedCrop(size=28, scale=(0.8, 1.0))
}

# To preview these augmentations, load a sample image from the training dataset.
# (Assuming the dataset returns a tensor; we convert it back to PIL for visualization.)
to_pil = transforms.ToPILImage()

# Load a sample image (first image from the training set)
train_transform_basic = transforms.Compose([transforms.ToTensor()])  # Basic transform (no augmentation)
sample_dataset = DataClass(split='train', transform=train_transform_basic, download=download)
sample_img_tensor, _ = sample_dataset[0]
sample_img = to_pil(sample_img_tensor)

# Display the sample image with each augmentation
plt.figure(figsize=(12, 6))
for i, (name, aug) in enumerate(augmentation_options.items()):
    # Create a composed transform with the augmentation followed by converting to tensor.
    transform = transforms.Compose([
        aug,
        transforms.ToTensor()
    ])
    augmented_tensor = transform(sample_img)
    augmented_img = to_pil(augmented_tensor)
    plt.subplot(2, 3, i+1)
    plt.imshow(augmented_img)
    plt.title(name)
    plt.axis("off")
plt.tight_layout()
plt.show()


# %%
# ----------------------------------------------
# Use Your Chosen Augmentation Pipeline for Training
# ----------------------------------------------

# For example, here we combine several augmentations:
# (You can change the order and parameters to see their effects on performance.)
chosen_train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),  # you can toggle this on/off
    transforms.RandomRotation(15),        # small rotation
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # adjust brightness and contrast
    transforms.Normalize(mean, std)
])

# For validation and testing, we generally only apply normalization.
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load the datasets with predefined splits: train, val, and test using the chosen transforms.
train_dataset = DataClass(split='train', transform=chosen_train_transform, download=download)
val_dataset = DataClass(split='val', transform=val_transform, download=download)
test_dataset = DataClass(split='test', transform=test_transform, download=download)

# Create DataLoaders for each split.
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Train dataset size:", len(train_dataset))
print("Validation dataset size:", len(val_dataset))
print("Test dataset size:", len(test_dataset))

# %%
# -----------------------
# CNN Architecture
# -----------------------

class CNN(nn.Module):
    def __init__(self, num_classes=num_classes, input_channels=3, image_size=28):
        """
        Args:
            num_classes (int): Number of classes in the dataset.
            input_channels (int): Number of input channels (3 for RGB images).
            image_size (int): The height/width of the input images (assumed square).
        """
        super(CNN, self).__init__()
        # First convolutional layer: 3-channel (RGB) to 32 feature maps
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        # Second convolutional layer: 32 to 64 feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Max pooling layer: reduces spatial dimensions by a factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the dimension of the features after two pooling layers (image_size/4)
        fc_input_dim = 64 * (image_size // 4) * (image_size // 4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolution -> ReLU -> Pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)
        # Fully connected layers with dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# %%
# -----------------------
# Model Initialization
# -----------------------

# Initialize the CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(num_classes=num_classes, input_channels=3, image_size=28).to(device)
print(model)


# %%
# -----------------------
# Training, Optimization, and Evaluation Example
# -----------------------

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs for training
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Training loop
    for inputs, labels in train_loader:
        # Move data to the appropriate device (GPU or CPU)
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.squeeze()  # Ensure labels are 1D (batch_size,)
        
        optimizer.zero_grad()       # Clear gradients
        outputs = model(inputs)     # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()             # Backpropagation
        optimizer.step()            # Update weights
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")
    
    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()  # Ensure labels are 1D
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")


# %% [markdown]
# ## 5. Data Augmentation, Training, Evaluation, and Comparison
# 
# In this section, we compare two training setups:
# - **With Data Augmentation:**  
#   The training data is augmented (using random horizontal flips) to introduce variability. This helps the network generalize better.
# - **Without Data Augmentation:**  
#   The training data is only normalized without any augmentation.
# 
# The steps are as follows:
# 
# 1. **Dataset Loading & Transforms:**  
#    - We define two different transform pipelines for the training set. One includes data augmentation (random horizontal flips) while the other does not.
#    - The validation and test sets use a common normalization transform.
#    - We load the PathMNIST dataset (a biomedical image dataset from MedMNIST) using the corresponding transforms for each experiment.
#   
# 2. **Model Architecture:**  
#    - We use the same CNN architecture that is adapted for 3-channel (RGB) images.
#   
# 3. **Training Function:**  
#    - A training function is defined to train the network for a number of epochs.
#    - It stores training and validation losses as well as accuracies at each epoch.
#   
# 4. **Evaluation:**  
#    - After training, we evaluate each model on the test set and compute a confusion matrix.
#   
# 5. **Visualization:**  
#    - We plot the training and validation loss and accuracy curves for both models.
#    - We also display the confusion matrices for the networks with and without augmentation.
#   
# This comprehensive implementation helps illustrate the impact of data augmentation on model performance.
# 
# The following code:
# 
# - Loads the PathMNIST biomedical dataset using separate transforms for training with and without augmentation.
# 
# - Trains two identical CNN models using the different training setups.
# 
# - Stores and prints the training and validation losses and accuracies.
# 
# - Evaluates the models on the test set and computes confusion matrices.
# 
# - Plots loss and accuracy curves along with confusion matrices to compare the effects of data augmentation.
# 

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# -----------------------
# Dataset and Transforms
# -----------------------

# Specify dataset parameters
data_flag = 'pathmnist'
download = True
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])
# Try to get number of classes; if missing, default to 9 (PathMNIST has 9 classes)
num_classes = info.get('n_classes', 9)
image_size = 28  # Adjust according to dataset

# Normalization parameters (for RGB images)
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# Define transform pipelines:
# With augmentation: includes random horizontal flip
train_transform_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean, std)
])

# Without augmentation: only normalization
train_transform_noaug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# For validation and testing, we use only normalization
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load training datasets for both experiments
train_dataset_aug = DataClass(split='train', transform=train_transform_aug, download=download)
train_dataset_noaug = DataClass(split='train', transform=train_transform_noaug, download=download)

# Load validation and test datasets (common for both)
val_dataset = DataClass(split='val', transform=val_transform, download=download)
test_dataset = DataClass(split='test', transform=test_transform, download=download)

# Create DataLoaders
batch_size = 32
train_loader_aug = DataLoader(train_dataset_aug, batch_size=batch_size, shuffle=True)
train_loader_noaug = DataLoader(train_dataset_noaug, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Train dataset sizes:", len(train_dataset_aug), "and", len(train_dataset_noaug))
print("Validation dataset size:", len(val_dataset))
print("Test dataset size:", len(test_dataset))

# %%
# -----------------------
# CNN Architecture Definition
# -----------------------

class CNN(nn.Module):
    def __init__(self, num_classes=num_classes, input_channels=3, image_size=image_size):
        """
        Args:
            num_classes (int): Number of classes.
            input_channels (int): Number of input channels (3 for RGB).
            image_size (int): Size (height/width) of the input images.
        """
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate input dimension for the fully connected layer
        fc_input_dim = 64 * (image_size // 4) * (image_size // 4)
        
        # Fully connected layers
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# %%
# -----------------------
# Training and Evaluation Functions
# -----------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()  # Ensure labels are 1D (batch_size,)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()
        
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation loop
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze()  # Ensure labels are 1D
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()
        
        val_loss = running_val_loss / total_val
        val_acc = correct_val / total_val
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs
    }

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()  # Ensure labels are 1D
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return all_preds, all_labels

# %%
# -----------------------
# Training: With and Without Data Augmentation
# -----------------------

num_epochs = 10

# Initialize two separate models (one for each experiment)
model_aug = CNN(num_classes=num_classes, input_channels=3, image_size=image_size).to(device)
model_noaug = CNN(num_classes=num_classes, input_channels=3, image_size=image_size).to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizers
optimizer_aug = optim.Adam(model_aug.parameters(), lr=0.001)
optimizer_noaug = optim.Adam(model_noaug.parameters(), lr=0.001)

print("Training model with data augmentation...")
metrics_aug = train_model(model_aug, train_loader_aug, val_loader, criterion, optimizer_aug, num_epochs, device)

print("\nTraining model without data augmentation...")
metrics_noaug = train_model(model_noaug, train_loader_noaug, val_loader, criterion, optimizer_noaug, num_epochs, device)

# -----------------------
# Evaluation on Test Set and Confusion Matrices
# -----------------------

# Evaluate both models on the test set
preds_aug, labels_aug = evaluate_model(model_aug, test_loader, device)
preds_noaug, labels_noaug = evaluate_model(model_noaug, test_loader, device)

cm_aug = confusion_matrix(labels_aug, preds_aug)
cm_noaug = confusion_matrix(labels_noaug, preds_noaug)

# %%
# -----------------------
# Visualization: Losses, Accuracies, and Confusion Matrices
# -----------------------

epochs = range(1, num_epochs+1)

# Plot Loss and Accuracy curves
plt.figure(figsize=(12,5))

# Loss curves
plt.subplot(1,2,1)
plt.plot(epochs, metrics_aug["train_losses"], label="Train Loss (Aug)")
plt.plot(epochs, metrics_aug["val_losses"], label="Val Loss (Aug)")
plt.plot(epochs, metrics_noaug["train_losses"], label="Train Loss (No Aug)")
plt.plot(epochs, metrics_noaug["val_losses"], label="Val Loss (No Aug)")
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy curves
plt.subplot(1,2,2)
plt.plot(epochs, metrics_aug["train_accs"], label="Train Acc (Aug)")
plt.plot(epochs, metrics_aug["val_accs"], label="Val Acc (Aug)")
plt.plot(epochs, metrics_noaug["train_accs"], label="Train Acc (No Aug)")
plt.plot(epochs, metrics_noaug["val_accs"], label="Val Acc (No Aug)")
plt.title("Accuracy Curves")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# Plot confusion matrices
plt.figure(figsize=(12,5))
tick_marks = np.arange(num_classes)

# Confusion matrix for model with augmentation
plt.subplot(1,2,1)
plt.imshow(cm_aug, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - With Augmentation")
plt.colorbar()
plt.xticks(tick_marks)
plt.yticks(tick_marks)
plt.xlabel("Predicted")
plt.ylabel("True")

# Confusion matrix for model without augmentation
plt.subplot(1,2,2)
plt.imshow(cm_noaug, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Without Augmentation")
plt.colorbar()
plt.xticks(tick_marks)
plt.yticks(tick_marks)
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.show()


# %% [markdown]
# As we could see on the charts, the augmentation must be used with care because it can improve but also worsen the results and the performance of the model!

# %% [markdown]
# ## 6. Transfer Learning Models & Comparison
# 
# In this section, we will:
# 
# 1. **Build custom transfer learning models** using several popular pre-trained networks:
#    - VGG16 (already demonstrated)
#    - VGG19
#    - ResNet50
#    - InceptionV3
#    - MobileNetV2
#    - EfficientNetB0
# 
# 2. **Prepare the dataset** for transfer learning:
#    - We use the PathMNIST biomedical dataset.
#    - Since the pre-trained models expect larger input images (e.g., 224×224), we resize the images and apply ImageNet normalization.
#    - We also include data augmentation (random horizontal flips) in the training transform.
# 
# 3. **Model Construction**:
#    - For each model, we load the base model with ImageNet weights (excluding the top classification layers).
#    - Freeze the base model layers.
#    - Add a custom classification head for our number of classes.
#    - Train the model on the same training data and validate on the same validation data.
#    - We store the trained models and their training/validation metrics in a dictionary for later evaluation.
# 
# 4. **Comparison & Visualization**:
#    - We plot accuracy and loss curves for each transfer learning model.
#    - We compute and display confusion matrices for test data.
# 
# 5. **Fine-Tuning**:
#    - From the above models, we select the three best (based on validation accuracy).
#    - For each, we unfreeze all layers (fine-tuning) and train further with a lower learning rate.
#    - Finally, we compare the fine-tuned results with the pre–fine-tuning results (using both training curves and confusion matrices).
# 
# For demonstration purposes, we use a small number of epochs. In practice, you may train longer or use early stopping.

# %% [markdown]
# The following code presents:
# 
# ### Dataset Preparation
# - We load the biomedical PathMNIST dataset using MedMNIST.
# - Because pre-trained networks expect larger images with ImageNet normalization, the images are resized to 256 and center cropped to 224.
# - Random horizontal flips are applied as augmentation in the training transform.
# 
# ### Transfer Learning Model Construction
# - A helper function `build_transfer_model` loads a pre-trained network (VGG16, VGG19, ResNet50, InceptionV3, MobileNetV2, or EfficientNetB0) with ImageNet weights.
# - The base layers are frozen (i.e., their weights are not updated during training) and a new classification head is attached to output the required number of classes.
# 
# ### Training and Evaluation
# - The `train_model` function trains each model for a set number of epochs, recording training and validation losses and accuracies.
# - The `evaluate_model` function computes predictions on the test set and builds a confusion matrix.
# - We iterate over the list of pre-trained models, train each one, and store their metrics and confusion matrices in the `trained_models` dictionary.
# - We then visualize the validation accuracy and loss curves as well as the confusion matrices for all models.
# 
# ### Fine-Tuning
# - The top three models (based on final validation accuracy) are selected.
# - For each, we unfreeze all layers and fine-tune with a lower learning rate.
# - Fine-tuning metrics and confusion matrices are recorded in the `fine_tuned_models` dictionary.
# - Finally, we compare pre–fine-tuning and post–fine-tuning performance via training curves and confusion matrices.
# 
# This complete code and explanation provide a comprehensive demonstration of transfer learning, performance comparison among various models, and the benefits of fine-tuning on the biomedical image dataset.
# 

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader
import medmnist
from medmnist import INFO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# -----------------------
# Dataset Preparation for Transfer Learning
# -----------------------

# Use the PathMNIST dataset
data_flag = 'pathmnist'
download = True
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])
# Use .get() to safely retrieve the number of classes; default to 9 if not found.
num_classes = info.get('n_classes', 9)

# For transfer learning, we use ImageNet normalization values and a larger image size.
# Setting input_size to 299 ensures that InceptionV3 receives the proper input size.
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]
input_size = 299  # Use 299 for all models so that InceptionV3 works properly

# Define transforms for training (with augmentation) and for validation/test.
train_transform_tl = transforms.Compose([
    transforms.Resize(330),                # Slightly larger to allow a good crop
    transforms.CenterCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])
val_test_transform_tl = transforms.Compose([
    transforms.Resize(330),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

# Load the dataset splits.
train_dataset_tl = DataClass(split='train', transform=train_transform_tl, download=download)
val_dataset_tl   = DataClass(split='val', transform=val_test_transform_tl, download=download)
test_dataset_tl  = DataClass(split='test', transform=val_test_transform_tl, download=download)

# Create DataLoaders.
batch_size = 32
train_loader_tl = DataLoader(train_dataset_tl, batch_size=batch_size, shuffle=True)
val_loader_tl   = DataLoader(val_dataset_tl, batch_size=batch_size, shuffle=False)
test_loader_tl  = DataLoader(test_dataset_tl, batch_size=batch_size, shuffle=False)

print("Training samples:", len(train_dataset_tl))
print("Validation samples:", len(val_dataset_tl))
print("Test samples:", len(test_dataset_tl))

# %%
# -----------------------
# Training and Evaluation Functions
# -----------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()  # Ensure labels are 1D
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()
        
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze()  # Ensure labels are 1D
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()
        
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}", flush=True)
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs
    }

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()  # Ensure labels are 1D
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return all_preds, all_labels


# %%
import torch.nn as nn
from torchvision import models
from torchvision.models import Inception_V3_Weights

# -----------------------
# Build Transfer Learning Models
# -----------------------

# Helper wrapper for InceptionV3
class InceptionWrapper(nn.Module):
    def __init__(self, inception_model):
        super(InceptionWrapper, self).__init__()
        self.inception = inception_model
    def forward(self, x):
        result = self.inception(x)
        # If the output is a tuple (e.g., (primary_output, aux_output)), return only the primary.
        if isinstance(result, tuple):
            return result[0]
        else:
            return result

def build_transfer_model(model_name, num_classes):
    """Load a pre-trained model, freeze its base layers, and add a custom classification head."""
    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif model_name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "inception_v3":
        # Load InceptionV3 with aux_logits=True as required by the weights.
        model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        # Wrap the model so that forward() returns only the primary output.
        model = InceptionWrapper(model)
    elif model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for param in model.features.parameters():
            param.requires_grad = False
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Model name not recognized.")
    return model

# List of transfer learning model names to train
model_names = ["vgg16", "vgg19", "resnet50", "inception_v3", "mobilenet_v2", "efficientnet_b0"]


# %%
# -----------------------
# Train Transfer Learning Models
# -----------------------

trained_models = {}
num_epochs_tl = 5  # For demonstration purposes
criterion = nn.CrossEntropyLoss()

for mname in model_names:
    print(f"\nTraining model: {mname}")
    model = build_transfer_model(mname, num_classes)
    model = model.to(device)
    # Only parameters with requires_grad=True are passed to the optimizer.
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    # Train the model and show progress using tqdm
    metrics = train_model(model, train_loader_tl, val_loader_tl, criterion, optimizer, num_epochs_tl, device)
    
    preds, labels = evaluate_model(model, test_loader_tl, device)
    cm = confusion_matrix(labels, preds)
    trained_models[mname] = {"model": model, "metrics": metrics, "confusion_matrix": cm}

# %%
# -----------------------
# Visualization: Training Curves for Transfer Learning Models
# -----------------------

plt.figure(figsize=(14,6))
for mname, results in trained_models.items():
    plt.plot(results["metrics"]["val_accs"], label=f"{mname} Val Acc")
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(14,6))
for mname, results in trained_models.items():
    plt.plot(results["metrics"]["val_losses"], label=f"{mname} Val Loss")
plt.title("Validation Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot confusion matrices for each model (on test set)
fig, axes = plt.subplots(2, 3, figsize=(16,10))
axes = axes.flatten()
for ax, (mname, results) in zip(axes, trained_models.items()):
    im = ax.imshow(results["confusion_matrix"], interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{mname} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()

# %%
import copy
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# Fine-Tuning the Top 3 Models
# -----------------------

# Select the top 3 models based on final validation accuracy
val_accs = {mname: results["metrics"]["val_accs"][-1] for mname, results in trained_models.items()}
sorted_models = sorted(val_accs.items(), key=lambda x: x[1], reverse=True)
top3 = [name for name, acc in sorted_models[:3]]
print("Top 3 models for fine-tuning:", top3)

fine_tuned_models = {}
num_epochs_ft = 3  # Fewer epochs for fine-tuning
for mname in top3:
    print(f"\nFine-tuning model: {mname}")
    # Get a fresh copy of the model from the trained_models (so we keep the pre-tuned version for comparison)
    model_ft = copy.deepcopy(trained_models[mname]["model"])
    # Unfreeze all layers
    for param in model_ft.parameters():
        param.requires_grad = True
    model_ft = model_ft.to(device)
    # Use a lower learning rate for fine-tuning
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=1e-4)
    metrics_ft = train_model(model_ft, train_loader_tl, val_loader_tl, criterion, optimizer_ft, num_epochs_ft, device)
    preds_ft, labels_ft = evaluate_model(model_ft, test_loader_tl, device)
    cm_ft = confusion_matrix(labels_ft, preds_ft)
    fine_tuned_models[mname] = {"model": model_ft, "metrics": metrics_ft, "confusion_matrix": cm_ft}


# %%
# -----------------------
# Visualization: Fine-Tuning Comparison
# -----------------------

# Compare validation accuracies and losses before and after fine-tuning for each top model
for mname in top3:
    plt.figure(figsize=(12,5))
    # Plot Accuracy
    plt.subplot(1,2,1)
    plt.plot(trained_models[mname]["metrics"]["val_accs"], label="Pre-Tuning")
    plt.plot(range(num_epochs_tl, num_epochs_tl+num_epochs_ft), fine_tuned_models[mname]["metrics"]["val_accs"], label="Fine-Tuned")
    plt.title(f"{mname} Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    # Plot Loss
    plt.subplot(1,2,2)
    plt.plot(trained_models[mname]["metrics"]["val_losses"], label="Pre-Tuning")
    plt.plot(range(num_epochs_tl, num_epochs_tl+num_epochs_ft), fine_tuned_models[mname]["metrics"]["val_losses"], label="Fine-Tuned")
    plt.title(f"{mname} Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Plot confusion matrices for pre-tuned vs. fine-tuned models (for test set)
fig, axes = plt.subplots(len(top3), 2, figsize=(10, 5*len(top3)))
if len(top3)==1:
    axes = np.expand_dims(axes, axis=0)
for i, mname in enumerate(top3):
    # Pre-tuning confusion matrix
    ax = axes[i,0]
    im = ax.imshow(trained_models[mname]["confusion_matrix"], interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{mname} Pre-Tuning CM")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    # Fine-tuned confusion matrix
    ax = axes[i,1]
    im = ax.imshow(fine_tuned_models[mname]["confusion_matrix"], interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(f"{mname} Fine-Tuned CM")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 7. Visualization Techniques and Visualizing Internal Representations Across Networks
# 
# In this section, we apply several visualization techniques to compare the internal quality of the trained filters and outputs from different transfer learning networks. We use the models stored from the transfer learning step on our biomedical (PathMNIST) dataset. The following visualizations are included:
# 
# - **Confusion Matrices:**  
#   Compare the classification performance on the validation (or test) set across networks.
#   We compute and plot confusion matrices on the validation set for each model to compare classification performance.
# 
# - **Filter Visualization:**  
#   Display the learned weights of the first convolutional layer to observe the low‐level features.
#   We display the weights of the first convolutional layer to reveal the low-level features learned by each network.
# 
# - **Activation Maps:**  
#   Show the output feature maps from the first convolutional layer for a sample image, revealing how the network processes the input.
#   We capture and visualize the activation maps from the first convolutional layer for a sample image, showing how each network processes the input.
# 
# - **Grad-CAM Heatmaps:**  
#   Compute and overlay Grad-CAM heatmaps on a sample image to highlight the class‐discriminative regions used by each model for decision making.
#   We compute Grad-CAM heatmaps using a designated target layer for each model, overlaid the heatmaps on the original sample image, and compared the class-discriminative regions highlighted by the networks.
# 
# Below, we also load a sample image (adjust the path as needed), preprocess it for input into the models, and retain a copy of the original image for overlaying the Grad-CAM heatmap.
# 
# This setup allows us to visually compare not only the overall performance (via confusion matrices) but also the internal representations (filters, activation maps, and Grad-CAM heatmaps) across different transfer learning models on our biomedical dataset. Adjust the target layers, sample image path, blending parameters, or colormaps as needed for deeper insights.

# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image

# %%
# -----------------------
# ----- Helper: Load and Preprocess Sample Image -----
# -----------------------

def load_and_preprocess_image(img_path, target_size=(224, 224), normalize=True):
    """
    Loads an image from disk, resizes it, and scales pixel values to [0,1].
    Returns both the preprocessed image (as a tensor) for model input and the original image as a NumPy array.
    """
    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize(target_size)
    original_img = np.array(img_resized)  # For visualization (range 0-255)
    
    # Preprocess: convert to tensor and normalize using ImageNet stats if required
    transform_list = [T.ToTensor()]
    if normalize:
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std  = [0.229, 0.224, 0.225]
        transform_list.append(T.Normalize(mean=imagenet_mean, std=imagenet_std))
    transform = T.Compose(transform_list)
    input_tensor = transform(img_resized).unsqueeze(0)  # Shape: (1, 3, H, W)
    
    return input_tensor.to(device), original_img

# Set the target size for the sample image.
img_rows, img_cols = 224, 224

# Specify the directory where sample images are stored.
sample_dir = os.path.join(os.getcwd(), 'data')

# Define valid image extensions.
valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

# List all files in the directory with a valid image extension.
all_files = [f for f in os.listdir(sample_dir) if os.path.splitext(f)[1].lower() in valid_extensions]

if not all_files:
    raise ValueError("No valid image files found in directory: " + sample_dir)

# Randomly choose one image file.
sample_img_name = np.random.choice(all_files)
sample_img_path = os.path.join(sample_dir, sample_img_name)
print("Randomly selected sample image:", sample_img_path)

# Load and preprocess the image.
#sample_img_path = os.path.join(os.getcwd(), 'data', 'sample_biomed.jpg')  # Adjust this path
img_tensor, original_img = load_and_preprocess_image(sample_img_path, target_size=(img_rows, img_cols))

# %%
import random
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage

# Assume test_dataset is already defined (from MedMNIST)
# Select a random index from the test dataset
rand_idx = random.randint(0, len(test_dataset) - 1)
sample_img_tensor, sample_label = test_dataset[rand_idx]

# The image is likely already normalized using the test_transform.
# To visualize it properly, we need to de-normalize it.
# Here we assume normalization was done using ImageNet stats:
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

# De-normalize the image
unnormalized_img = sample_img_tensor * imagenet_std + imagenet_mean
# Clamp values to [0,1] for proper display
unnormalized_img = torch.clamp(unnormalized_img, 0, 1)

# Convert the tensor to a PIL image for visualization
to_pil = ToPILImage()
sample_img = to_pil(unnormalized_img)

# Display the image using matplotlib
plt.figure(figsize=(5,5))
plt.imshow(sample_img)
plt.title(f"Random Test Image (Label: {sample_label})")
plt.axis("off")
plt.show()


# %% [markdown]
# ## Visualize comparisons of model focus areas using heatmaps
# 
# 1. **Selecting a Random Test Image:**
#    - The `select_random_test_image` function randomly chooses an image from the `test_dataset`, adds a batch dimension to create an input tensor, and de-normalizes the image (assuming ImageNet normalization) for proper visualization.
# 
# 2. **Grad-CAM Helper Functions:**
#    - `generate_gradcam`: Registers forward and backward hooks on the chosen target layer, computes the gradients of the predicted class, and produces a heatmap by averaging weighted activations.
#    - `overlay_heatmap_on_image`: Resizes the heatmap to match the original image, applies a colormap (e.g., `cv2.COLORMAP_JET`), and overlays it on the original image with a specified blending factor.
# 
# 3. **Retrieving the Target Layer:**
#    - The `get_target_layer` function returns a suitable convolutional layer from each model based on the model name. (For instance, for VGG16, the last conv layer in `model.features` is used.)
# 
# 4. **Main Comparison Section:**
#    - The code randomly selects a test image from the `test_dataset` and then iterates over the top fine-tuned models stored in `fine_tuned_models`.
#    - For each model, it retrieves the appropriate target layer, computes the Grad-CAM heatmap, overlays it on the original image, and displays the results side-by-side.
#    - The final plot shows a graphical comparison of the Grad-CAM heatmaps, illustrating which regions each model focuses on when classifying the test image.
# 
# This combined code allows you and your students to visually compare the reaction of the top three fine-tuned transfer learning models to the same test image, offering insights into the areas of the image that drive each model’s prediction.
# 
# 
# Below is an example that randomly selects a test image from the test dataset, computes Grad-CAM heatmaps for each of the three top (fine-tuned) transfer learning models, and then displays a graphical comparison of the heatmap overlays. In this example, we assume that your fine-tuned models are stored in a dictionary called `fine_tuned_models` with keys such as `"vgg16"`, `"resnet50"`, etc.
# 
# The code includes helper functions to:
# - Randomly select and de-normalize a test image.
# - Compute a Grad-CAM heatmap given a model, its target layer, and an input tensor.
# - Overlay the heatmap on the original image.
# - Retrieve a target layer from each model (adjust this function if your model architecture differs).
# 
# Below is the complete code:
# 

# %%
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as T
from torchvision.transforms import ToPILImage

# -----------------------
# Helper: Randomly select and de-normalize a test image
# -----------------------

def select_random_test_image(test_dataset, target_size=(224, 224)):
    """
    Randomly selects an image from the test_dataset, returns both:
      - input_tensor: properly shaped for the model (with batch dimension).
      - original_img: a NumPy array (H, W, 3) de-normalized for visualization.
    
    Assumes test_dataset images were normalized using ImageNet stats.
    """
    # Randomly choose an index
    rand_idx = random.randint(0, len(test_dataset) - 1)
    sample_img_tensor, sample_label = test_dataset[rand_idx]
    
    # Prepare the tensor for model input (add batch dimension)
    input_tensor = sample_img_tensor.unsqueeze(0).to(device)
    
    # De-normalize for visualization (assuming ImageNet normalization)
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    img_np = sample_img_tensor.cpu().numpy()  # shape: (C, H, W)
    for i in range(3):
        img_np[i] = img_np[i] * imagenet_std[i] + imagenet_mean[i]
    img_np = np.clip(img_np, 0, 1)
    img_np = np.transpose(img_np, (1, 2, 0))  # convert to HWC format
    img_np = np.uint8(img_np * 255)
    
    return input_tensor, img_np, sample_label

# -----------------------
# Helper: Grad-CAM Functions
# -----------------------

def generate_gradcam(model, input_tensor, target_layer):
    """
    Computes the Grad-CAM heatmap for a given model and input_tensor using the specified target_layer.
    """
    model.eval()
    gradients = None
    activations = None

    # Define hook functions to capture gradients and activations.
    def save_gradient(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    def save_activation(module, input, output):
        nonlocal activations
        activations = output

    # Register hooks on the target layer.
    hook_a = target_layer.register_forward_hook(save_activation)
    hook_g = target_layer.register_backward_hook(save_gradient)
    
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()
    
    # Remove hooks.
    hook_a.remove()
    hook_g.remove()
    
    # Global average pooling of the gradients.
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # Weight the activations by the pooled gradients.
    activations = activations[0]
    for i in range(activations.shape[0]):
        activations[i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)
    return heatmap

def overlay_heatmap_on_image(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays the Grad-CAM heatmap on the original image.
    img: original image as a NumPy array (H, W, 3) in BGR or RGB (values 0-255).
    heatmap: Grad-CAM heatmap as a 2D numpy array (values between 0 and 1).
    """
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# -----------------------
# Helper: Retrieve Target Layer from Model
# -----------------------

def get_target_layer(model, model_name):
    """
    Returns an appropriate target layer (a convolutional layer) from the given model.
    Adjust the returned layer according to your model architecture.
    """
    if model_name.lower() == "vgg16":
        return model.features[29]  # Last conv layer in VGG16
    elif model_name.lower() == "vgg19":
        return model.features[35]  # Last conv layer in VGG19
    elif model_name.lower() == "resnet50":
        return model.layer4[-1]    # Last block of ResNet50
    elif model_name.lower() == "inception_v3":
        # In our wrapper, we have model.inception; choose "Mixed_7c" if available.
        for name, module in model.inception.named_modules():
            if name == "Mixed_7c":
                return module
        # Otherwise fallback to the last child
        return list(model.inception.children())[-1]
    elif model_name.lower() == "mobilenet_v2":
        return model.features[18]  # Last conv layer in MobileNetV2
    elif model_name.lower() == "efficientnet_b0":
        return model.features[-1]    # Last conv layer in EfficientNetB0
    else:
        raise ValueError("Model name not recognized")

# -----------------------
# Main: Generate and Compare Grad-CAM Heatmaps for Top 3 Fine-Tuned Models
# -----------------------

# Assume fine_tuned_models is a dictionary with keys for the top 3 models, e.g.:
# fine_tuned_models = {
#    "vgg16": {"model": <model>, "metrics": {...}, "confusion_matrix": ... },
#    "resnet50": {...},
#    "inception_v3": {...}
# }

# Randomly select a test image from test_dataset.
input_tensor, original_img, sample_label = select_random_test_image(test_dataset, target_size=(224, 224))
print("Selected test image label:", sample_label)

# Create a figure to display the Grad-CAM overlays from each model.
num_models = len(fine_tuned_models)
plt.figure(figsize=(5 * num_models, 5))

for i, (mname, model_data) in enumerate(fine_tuned_models.items()):
    model_ft = model_data["model"]
    # Retrieve the target layer for the current model.
    target_layer = get_target_layer(model_ft, mname)
    # Compute the Grad-CAM heatmap.
    heatmap = generate_gradcam(model_ft, input_tensor, target_layer)
    # Overlay the heatmap on the original image.
    overlay_img = overlay_heatmap_on_image(original_img, heatmap, alpha=0.4)
    # Convert color from BGR to RGB for display with matplotlib.
    overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, num_models, i + 1)
    plt.imshow(overlay_rgb)
    plt.title(f"{mname} Grad-CAM")
    plt.axis("off")
    
plt.tight_layout()
plt.show()


# %% [markdown]
# # Explanation
# 
# 1. **Selecting a Random Test Image:**  
#    The `select_random_test_image` function randomly selects an image from the `test_dataset`, adds a batch dimension to create an input tensor, and de-normalizes the image (assuming ImageNet normalization) to produce a visualization-ready NumPy array.
# 
# 2. **Grad-CAM Functions:**  
#    - `generate_gradcam` registers forward and backward hooks on the specified target layer, computes the gradients of the predicted class, weights the activations accordingly, and produces a normalized heatmap.
#    - `overlay_heatmap_on_image` resizes the heatmap to match the original image, applies a colormap, and blends it with the original image.
# 
# 3. **Retrieving the Target Layer:**  
#    The `get_target_layer` function returns an appropriate convolutional layer from each model based on its architecture (e.g., the last convolutional layer of VGG16, the final block of ResNet50, etc.). Adjust this function as needed for your specific models.
# 
# 4. **Main Comparison Section:**  
#    The code randomly selects a test image from the test dataset and then iterates over the fine-tuned models stored in `fine_tuned_models`. For each model, it:
#    - Retrieves the appropriate target layer.
#    - Computes the Grad-CAM heatmap.
#    - Overlays the heatmap on the original image.
#    - Displays the results side-by-side for comparison.
# 
# This setup allows you and your students to visually compare the regions each top fine-tuned model focuses on when classifying the same test image.
# 

# %%
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as T
from torchvision.transforms import ToPILImage

# Assume that the device is already set (e.g., device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# and that test_dataset is defined.

# -----------------------
# Helper: Randomly Select and De-Normalize a Test Image
# -----------------------
def select_random_test_image(test_dataset, target_size=(224, 224)):
    """
    Randomly selects an image from the test_dataset and returns:
      - input_tensor: the image tensor with a batch dimension, ready for model input.
      - original_img: the de-normalized image as a NumPy array (H, W, 3) for visualization.
      - sample_label: the ground-truth label of the image.
    
    Assumes that test_dataset images were normalized using ImageNet statistics.
    """
    # Randomly choose an index from the test dataset.
    rand_idx = random.randint(0, len(test_dataset) - 1)
    sample_img_tensor, sample_label = test_dataset[rand_idx]
    
    # Prepare the input tensor (add batch dimension).
    input_tensor = sample_img_tensor.unsqueeze(0).to(device)
    
    # De-normalize the image for visualization.
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std  = np.array([0.229, 0.224, 0.225])
    
    img_np = sample_img_tensor.cpu().numpy()  # Shape: (C, H, W)
    for i in range(3):
        img_np[i] = img_np[i] * imagenet_std[i] + imagenet_mean[i]
    img_np = np.clip(img_np, 0, 1)
    img_np = np.transpose(img_np, (1, 2, 0))  # Convert to H x W x C format.
    original_img = np.uint8(img_np * 255)
    
    return input_tensor, original_img, sample_label

# -----------------------
# Helper: Generate Grad-CAM Heatmap
# -----------------------
def generate_gradcam(model, input_tensor, target_layer):
    """
    Computes the Grad-CAM heatmap for a given model and input_tensor using the specified target_layer.
    """
    model.eval()
    gradients = None
    activations = None

    def save_gradient(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]

    def save_activation(module, input, output):
        nonlocal activations
        activations = output

    # Register forward and backward hooks.
    hook_a = target_layer.register_forward_hook(save_activation)
    hook_g = target_layer.register_backward_hook(save_gradient)
    
    # Forward pass.
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()
    
    # Remove hooks.
    hook_a.remove()
    hook_g.remove()
    
    # Global average pooling of gradients.
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    # Weight the activations by the pooled gradients.
    activations = activations[0]
    for i in range(activations.shape[0]):
        activations[i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)
    return heatmap

# -----------------------
# Helper: Overlay Heatmap on Image
# -----------------------
def overlay_heatmap_on_image(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays the Grad-CAM heatmap on the original image.
    
    Parameters:
      img: the original image as a NumPy array (H, W, 3) with values in 0-255.
      heatmap: the Grad-CAM heatmap (values between 0 and 1).
      alpha: blending factor.
      colormap: OpenCV colormap (default: cv2.COLORMAP_JET).
    """
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# -----------------------
# Helper: Retrieve Target Layer from Model
# -----------------------
def get_target_layer(model, model_name):
    """
    Returns a suitable target layer (typically a convolutional layer) from the given model.
    Adjust this function if your model architectures differ.
    """
    if model_name.lower() == "vgg16":
        return model.features[29]  # Last conv layer in VGG16.
    elif model_name.lower() == "vgg19":
        return model.features[35]  # Last conv layer in VGG19.
    elif model_name.lower() == "resnet50":
        return model.layer4[-1]    # Last block of ResNet50.
    elif model_name.lower() == "inception_v3":
        # Look for a layer named "Mixed_7c".
        for name, module in model.inception.named_modules():
            if name == "Mixed_7c":
                return module
        return list(model.inception.children())[-1]
    elif model_name.lower() == "mobilenet_v2":
        return model.features[18]  # Last conv layer in MobileNetV2.
    elif model_name.lower() == "efficientnet_b0":
        return model.features[-1]  # Last conv layer in EfficientNetB0.
    else:
        raise ValueError("Model name not recognized")

# -----------------------
# Main: Generate and Compare Grad-CAM Heatmaps for Top 3 Fine-Tuned Models
# -----------------------

# Assume fine_tuned_models is a dictionary with keys for the top models (e.g., "vgg16", "resnet50", "inception_v3")
input_tensor, original_img, sample_label = select_random_test_image(test_dataset, target_size=(224, 224))
print("Selected test image label:", sample_label)

num_models = len(fine_tuned_models)
plt.figure(figsize=(5 * num_models, 5))
for i, (mname, model_data) in enumerate(fine_tuned_models.items()):
    model_ft = model_data["model"]
    target_layer = get_target_layer(model_ft, mname)
    heatmap = generate_gradcam(model_ft, input_tensor, target_layer)
    overlay_img = overlay_heatmap_on_image(original_img, heatmap, alpha=0.4)
    # Convert BGR to RGB for display.
    overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, num_models, i + 1)
    plt.imshow(overlay_rgb)
    plt.title(f"{mname} Grad-CAM")
    plt.axis("off")
plt.tight_layout()
plt.show()


# %%
# -----------------------
# ----- 5.1. Plotting Confusion Matrices -----
# -----------------------

def plot_confusion_matrix(model, model_name, data_loader, num_classes):
    """
    Computes predictions on the given data_loader and plots the confusion matrix.
    Works for multi-class classification.
    """
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Example usage: loop through trained models (assumed to be stored in the dictionary 'trained_models')
if 'trained_models' in globals():
    for model_name, model_data in trained_models.items():
        print(f"Confusion Matrix for {model_name}:")
        # Here, we use the validation DataLoader from transfer learning step (data loader variable: val_loader_tl)
        plot_confusion_matrix(model_data["model"], model_name, val_loader_tl, num_classes)
else:
    print("Error: 'trained_models' dictionary not found. Please run the transfer learning training cells first.")


# %%
# -----------------------
# ----- 5.2. Visualizing Filters of the First Convolutional Layer -----
# -----------------------

def visualize_filters(model, model_name, n_filters=6):
    """
    Visualizes the weights of the first convolutional layer in the model.
    Works for PyTorch models by searching for the first nn.Conv2d layer.
    """
    first_conv = None
    # Check if the model has a 'features' attribute (e.g., VGG, MobileNet, EfficientNet)
    if hasattr(model, 'features'):
        # Iterate in reverse order over features to find the first Conv2d layer
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                first_conv = layer
                break
    # For ResNet and Inception models, iterate over modules
    if first_conv is None:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                first_conv = module
                break
    if first_conv is None:
        print(f"No Conv2d layer found for {model_name}.")
        return

    # Get the filters (weights) and convert to CPU numpy array
    filters = first_conv.weight.data.cpu().numpy()  # Shape: (out_channels, in_channels, H, W)
    n_filters = min(n_filters, filters.shape[0])
    
    fig, axes = plt.subplots(1, n_filters, figsize=(20, 8))
    for i in range(n_filters):
        f = filters[i]
        # Normalize filter values to 0-1
        f_min, f_max = f.min(), f.max()
        f_norm = (f - f_min) / (f_max - f_min + 1e-8)
        # For visualization, if there are at least 3 input channels, take the first three channels
        if f_norm.shape[0] >= 3:
            img_to_show = np.transpose(f_norm[:3, :, :], (1, 2, 0))
        else:
            img_to_show = f_norm[0]
        axes[i].imshow(img_to_show)
        axes[i].axis('off')
        axes[i].set_title(f"{model_name}\nFilter {i+1}")
    plt.show()

# Example usage:
if 'trained_models' in globals():
    for model_name, model_data in trained_models.items():
        print(f"Visualizing first conv filters for {model_name}:")
        visualize_filters(model_data["model"], model_name)
else:
    print("Error: 'trained_models' dictionary not found. Please run the training cells first.")


# %% [markdown]
# Below is an example that randomly selects a test image from the test dataset, resizes it to a specified target size, and then uses it to display activation maps (via a forward hook on the first convolutional layer) for each trained model. This ensures that each model receives an input of the proper size. 
# 
# The code includes helper functions to:
# - Randomly select a test image and resize it to the desired target size.
# - De-normalize the image for visualization.
# - Retrieve the first convolutional layer and display its activation maps.
# - In the main section, it loops over the trained models (stored in `trained_models`) and shows a graphical comparison of the activation maps.
# 
# Below is the complete updated code:

# %%
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import ToPILImage

# -----------------------
# Helper: Randomly Select and Resize a Test Image
# -----------------------
def select_random_test_image_resized(test_dataset, target_size=(224, 224)):
    """
    Randomly selects an image from the test_dataset, resizes it to target_size,
    and returns:
      - input_tensor: the resized image tensor (with batch dimension) normalized for model input.
      - original_img: the de-normalized image as a NumPy array (H, W, 3) for visualization.
      - sample_label: the ground-truth label.
    
    This function bypasses the test_dataset transform and applies a custom transform
    so that the image can be resized to the desired target size.
    """
    # Randomly choose an index.
    rand_idx = random.randint(0, len(test_dataset) - 1)
    sample_img_tensor, sample_label = test_dataset[rand_idx]
    
    # Convert the tensor to a PIL image.
    to_pil = ToPILImage()
    pil_img = to_pil(sample_img_tensor.cpu())
    
    # Resize the image to the target size.
    pil_img_resized = pil_img.resize(target_size)
    
    # Define a transform: convert to tensor and normalize using ImageNet stats.
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    resized_tensor = transform(pil_img_resized)
    
    # For visualization: de-normalize the image.
    img_np = resized_tensor.cpu().numpy()  # shape: (C, H, W)
    for i in range(3):
        img_np[i] = img_np[i] * imagenet_std[i] + imagenet_mean[i]
    img_np = np.clip(img_np, 0, 1)
    img_np = np.transpose(img_np, (1, 2, 0))  # Convert to H x W x C
    original_img = np.uint8(img_np * 255)
    
    # Prepare the tensor for model input.
    input_tensor = resized_tensor.unsqueeze(0).to(device)
    
    return input_tensor, original_img, sample_label

# -----------------------
# Helper: Retrieve First Conv Layer and Display Activation Maps
# -----------------------
def get_first_conv_layer(model):
    """
    Returns the first Conv2d layer found in the model.
    """
    if hasattr(model, 'features'):
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                return layer
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            return module
    return None

def display_activation_maps(model, input_tensor, model_name):
    """
    Registers a forward hook on the first convolutional layer, performs a forward pass,
    and displays the resulting activation maps.
    """
    first_conv = get_first_conv_layer(model)
    if first_conv is None:
        print(f"No Conv2d layer found for {model_name}.")
        return
    
    activation = None
    def hook_fn(module, input, output):
        nonlocal activation
        activation = output.detach().cpu()
    hook_handle = first_conv.register_forward_hook(hook_fn)
    
    # Forward pass
    model.eval()
    _ = model(input_tensor)
    hook_handle.remove()
    
    if activation is None:
        print("Activation not captured.")
        return

    # Activation shape: (1, channels, H, W)
    activations = activation[0]
    num_features = activations.shape[0]
    cols = int(np.sqrt(num_features))
    rows = int(np.ceil(num_features / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = np.array(axes).reshape(-1)
    for i in range(num_features):
        ax = axes[i]
        act_map = activations[i].numpy()
        ax.imshow(act_map, cmap='viridis')
        ax.axis('off')
        ax.set_title(f"F{i+1}")
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.suptitle(f"Activation Maps from First Conv Layer ({model_name})")
    plt.tight_layout()
    plt.show()

# -----------------------
# Main: Visualize Activation Maps for Top Fine-Tuned Models
# -----------------------

# Assume that 'trained_models' and 'test_dataset' are defined.
if 'trained_models' in globals() and 'test_dataset' in globals():
    # Loop over each model in fine-tuned models.
    for model_name, model_data in fine_tuned_models.items():
        # Set target size based on model type: InceptionV3 uses 299x299, others use 224x224.
        if model_name.lower() == "inception_v3":
            target_size = (299, 299)
        else:
            target_size = (224, 224)
        print(f"\nDisplaying activation maps for {model_name} with input size {target_size}:")
        input_tensor, original_img, sample_label = select_random_test_image_resized(test_dataset, target_size=target_size)
        display_activation_maps(model_data["model"], input_tensor, model_name)
else:
    print("Error: 'trained_models' or 'test_dataset' is not defined. Please run the training and dataset loading cells first.")


# %% [markdown]
# Below is an example that randomly selects a test image from the test dataset (resized to the appropriate target size for each model), computes Grad-CAM heatmaps for each trained model, and displays a graphical comparison of the heatmap overlays. In this code, we call the helper function `select_random_test_image_resized` within the loop so that the necessary variables (`input_tensor` and `original_img`) are defined for each model's forward pass.
# 
# The code includes helper functions to:
# - Randomly select and resize a test image from the dataset.
# - Compute the Grad-CAM heatmap given a model, its target layer, and an input tensor.
# - Overlay the heatmap on the original image.
# - Retrieve a target layer from each model.
# 
# Below is the complete updated code:

# %%
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import ToPILImage

# -----------------------
# Helper: Randomly Select and Resize a Test Image
# -----------------------
def select_random_test_image_resized(test_dataset, target_size=(224, 224)):
    """
    Randomly selects an image from the test_dataset, resizes it to target_size,
    and returns:
      - input_tensor: the resized image tensor (with batch dimension) normalized for model input.
      - original_img: the de-normalized image as a NumPy array (H, W, 3) for visualization.
      - sample_label: the ground-truth label.
    
    This function bypasses the test_dataset transform and applies a custom transform
    so that the image can be resized to the desired target size.
    """
    # Randomly choose an index.
    rand_idx = random.randint(0, len(test_dataset) - 1)
    sample_img_tensor, sample_label = test_dataset[rand_idx]
    
    # Convert the tensor to a PIL image.
    to_pil = ToPILImage()
    pil_img = to_pil(sample_img_tensor.cpu())
    
    # Resize the image to the target size.
    pil_img_resized = pil_img.resize(target_size)
    
    # Define a transform: convert to tensor and normalize using ImageNet stats.
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
    resized_tensor = transform(pil_img_resized)
    
    # For visualization: de-normalize the image.
    img_np = resized_tensor.cpu().numpy()  # shape: (C, H, W)
    for i in range(3):
        img_np[i] = img_np[i] * imagenet_std[i] + imagenet_mean[i]
    img_np = np.clip(img_np, 0, 1)
    img_np = np.transpose(img_np, (1, 2, 0))  # convert to H x W x C format
    original_img = np.uint8(img_np * 255)
    
    # Prepare the tensor for model input (add batch dimension).
    input_tensor = resized_tensor.unsqueeze(0).to(device)
    
    return input_tensor, original_img, sample_label

# -----------------------
# Helper: Generate Grad-CAM Heatmap
# -----------------------
def generate_gradcam(model, input_tensor, target_layer):
    """
    Computes the Grad-CAM heatmap for a given model and input_tensor using the specified target_layer.
    """
    model.eval()
    activation = None
    gradient = None

    def forward_hook(module, input, output):
        nonlocal activation
        activation = output.detach()
    def backward_hook(module, grad_in, grad_out):
        nonlocal gradient
        gradient = grad_out[0].detach()
    
    hook_forward = target_layer.register_forward_hook(forward_hook)
    hook_backward = target_layer.register_backward_hook(backward_hook)
    
    # Forward pass
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    
    # Backward pass: compute gradients for the predicted class.
    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()
    
    hook_forward.remove()
    hook_backward.remove()
    
    # Global average pooling on gradients.
    weights = torch.mean(gradient, dim=(2,3), keepdim=True)
    grad_cam_map = torch.sum(weights * activation, dim=1, keepdim=True)
    grad_cam_map = torch.relu(grad_cam_map)
    # Normalize heatmap
    heatmap = grad_cam_map.squeeze().cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap

# -----------------------
# Helper: Overlay Heatmap on Image
# -----------------------
def overlay_heatmap_on_image(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays the Grad-CAM heatmap on the original image.
    img: original image as a NumPy array (H x W x 3) with values in 0-255.
    heatmap: Grad-CAM heatmap as a 2D numpy array (values between 0 and 1).
    """
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_resized, colormap)
    if img.dtype != np.uint8:
        img_uint8 = np.uint8(img)
    else:
        img_uint8 = img.copy()
    overlay = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

# -----------------------
# Helper: Retrieve Target Layer from Model
# -----------------------
def get_target_layer(model, model_name):
    """
    Returns a target layer for Grad-CAM based on the model type.
    For models with a 'features' attribute, returns the last Conv2d layer in features.
    For ResNet50, returns the last block of layer4.
    For InceptionV3, attempts to return a specific convolutional block.
    """
    if model_name in ["vgg16", "vgg19", "mobilenet_v2", "efficientnet_b0"]:
        if hasattr(model, 'features'):
            for layer in reversed(model.features):
                if isinstance(layer, nn.Conv2d):
                    return layer
    elif model_name == "resnet50":
        return model.layer4[-1]
    elif model_name == "inception_v3":
        for name, module in model.named_modules():
            if "Mixed_7c" in name:
                return module
    # Fallback: return the first Conv2d layer.
    return get_first_conv_layer(model)

# -----------------------
# Main: Generate and Compare Grad-CAM Heatmaps for Fine-Tuned Models
# -----------------------

if 'trained_models' in globals() and 'test_dataset' in globals():
    num_models = len(fine_tuned_models)
    plt.figure(figsize=(15, 10))
    for i, (model_name, model_data) in enumerate(fine_tuned_models.items()):
        # Set target size based on model type.
        if model_name.lower() == "inception_v3":
            target_size = (299, 299)
        else:
            target_size = (224, 224)
        print(f"Processing {model_name} with input size {target_size}")
        # Randomly select and resize a test image.
        input_tensor, original_img, sample_label = select_random_test_image_resized(test_dataset, target_size=target_size)
        # Retrieve target layer for Grad-CAM.
        target_layer = get_target_layer(model_data["model"], model_name.lower())
        if target_layer is None:
            print(f"Target layer not found for {model_name}. Skipping Grad-CAM.")
            continue
        # Generate the Grad-CAM heatmap.
        heatmap = generate_gradcam(model_data["model"], input_tensor, target_layer)
        # Overlay heatmap on original image.
        overlay_img = overlay_heatmap_on_image(original_img, heatmap)
        # Convert color from BGR to RGB for display.
        overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 3, i + 1)
        plt.imshow(overlay_rgb)
        plt.title(model_name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("Error: 'trained_models' dictionary not found. Please run the training cells first.")


# %% [markdown]
# # Assignments and Experiments
# 
# This notebook demonstrates:
# - **CNN Training:** Building and training a convolutional network from scratch.
# - **Data Augmentation:** Improving model generalization using augmentation techniques.
# - **Transfer Learning:** Leveraging a pre-trained VGG16 model (and others) to fine-tune on your dataset.
# - **Visualization:** Analyzing training performance and inspecting intermediate CNN representations and filters.
# 
# ## Experiment Assignments
# 
# 1. **Model Architecture Exploration:**
#    - **Objective:** Modify the model structure by varying:
#      - The number of convolutional layers.
#      - The number of filters per layer.
#      - Different kernel sizes.
#      - Various activation functions (ReLU, LeakyReLU, ELU, etc.).
#      - Adding or removing skip-connections.
#    - **Task:** Run experiments with different architecture configurations and compare their training/validation loss and accuracy curves.
# 
# 2. **Training Hyperparameters Tuning:**
#    - **Objective:** Optimize training hyperparameters such as:
#      - Batch size.
#      - Learning rate.
#      - Choice of optimizer (Adam, SGD, RMSprop).
#      - Number of epochs.
#    - **Task:** Run experiments varying one or more training hyperparameters and observe their effect on convergence and final performance.
# 
# 3. **Regularization Techniques Comparison:**
#    - **Objective:** Compare different regularization methods:
#      - Dropout (varying dropout rates).
#      - L1 and L2 regularization (weight decay).
#      - Data augmentation strategies.
#    - **Task:** Experiment with each regularization method (or combinations) to evaluate their impact on overfitting and performance.
# 
# 4. **Data Augmentation Variations:**
#    - **Objective:** Explore various augmentation functions (e.g., flips, rotations, color jitter, affine transformations) to understand their effect on model generalization.
#    - **Task:** Modify and combine augmentation functions. Visualize augmented images and compare model results when training with different augmentation schemes.
# 
# 5. **Comparison and Visualization:**
#    - **Objective:** Compare the experiments graphically.
#    - **Task:** Use provided plotting functions to display loss and accuracy curves for different experiments side-by-side. This will help you decide which hyperparameter settings yield the best performance.
# 
# Below, we provide example code that defines functions to build a custom CNN based on hyperparameters, run experiments, and graphically compare the results. Adapt it or write your code.

# %% [markdown]
# ### You can follow this workflow plan:
# 
# 1. **Model and Training Setup:**
#    - Use the `build_custom_cnn()` function (or your transfer learning models) to create a model with your chosen hyperparameters.
#    - Prepare your dataset and DataLoaders as shown in previous sections.
#    - Choose your training hyperparameters (learning rate, batch size, number of epochs, etc.) and define your optimizer and loss function.
# 
# 2. **Running Experiments:**
#    - Call `train_model()` for each experiment configuration. For example, you might vary the number of layers, filters, activation functions, or dropout rates.
#    - Save the returned metrics (loss and accuracy curves) in a dictionary. Use experiment names (e.g., "Exp1", "Exp2", etc.) as keys.
# 
# 3. **Evaluation:**
#    - Use `evaluate_model()` to obtain predictions and compute confusion matrices for each model on the test set.
# 
# 4. **Graphical Comparison:**
#    - Call `plot_experiments_results(experiments)` by passing your experiments dictionary. This will generate side-by-side plots for training and validation loss and accuracy curves, making it easy to compare the effects of different hyperparameter settings.
# 
# 5. **Exploration:**
#    - Modify the hyperparameters, the model architecture (using `build_custom_cnn()`), or training parameters. Rerun the experiments to observe their impact on performance.
#    - Experiment with regularization techniques (such as L1/L2 weight decay, dropout, and data augmentation) by adjusting the parameters in your model builder and training loop.
# 
# By following these steps, you can systematically study how changes in the network architecture, training hyperparameters, and regularization strategies affect model performance. The visual comparisons will help you understand the trade-offs and guide you toward the optimal configuration for your dataset.
# 
# **Dataset and Transforms:**
# 
# - The dataset transforms now include `transforms.Resize((224,224))`, ensuring that all images are resized to 224×224 regardless of their original aspect ratio.
# - The hyperparameters (in `hparams1` and `hparams2`) now specify `'image_size': 224`, so the model’s expected flattened feature size is computed as:
#   
#   \[
#   \text{fc\_input\_dim} = \text{filters}[-1] \times \left(\frac{224}{2^{\text{num\_conv\_layers}}}\right)^2.
#   \]
#   
#   For example, for a 2-layer model with filters [32,64]:
#   
#   \[
#   64 \times \left(\frac{224}{4}\right)^2 = 64 \times 56 \times 56 = 200704.
#   \]
#   
#   This ensures the fully connected layer is built with the correct input size.
# 
# **Custom CNN Builder:**
# 
# - The `build_custom_cnn` function builds a sequential model using the provided hyperparameters. It uses the given `image_size` to compute the flattened dimension for the FC layers.
# 
# **Training and Evaluation:**
# 
# - The `train_model` and `evaluate_model` functions remain unchanged except that they ensure that target labels are squeezed to one dimension.
# 
# **Plotting Results:**
# 
# - The `plot_experiments_results` function graphically compares training and validation loss and accuracy curves from different experiments.
# 
# **Training Experiments:**
# 
# - Two experiments are run: one with a baseline architecture (`hparams1`) and one with a modified architecture (`hparams2`

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import copy

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Dataset Loading & Preprocessing
# -----------------------

# Specify which MedMNIST dataset to use (here: 'pathmnist' for histopathologic images)
data_flag = 'pathmnist'
download = True  # Download if not available locally
info = INFO[data_flag]
DataClass = getattr(medmnist.dataset, info['python_class'])
num_classes = info.get('n_classes', 9)  # Default to 9 if key is missing

# Set the target image size (we choose 224x224)
image_size = 224

# Define normalization parameters (assumed RGB images)
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# **Important:** Use a tuple for Resize to force exact dimensions.
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean, std)
])
val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Load datasets using predefined splits provided by MedMNIST
train_dataset = DataClass(split='train', transform=train_transform, download=download)
val_dataset = DataClass(split='val', transform=val_transform, download=download)
test_dataset = DataClass(split='test', transform=test_transform, download=download)

# Create dataloaders for each split
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("Dataset loaded: {} training images, {} validation images, {} test images".format(
    len(train_dataset), len(val_dataset), len(test_dataset)
))

# -----------------------
# Custom CNN Builder
# -----------------------

def build_custom_cnn(hparams):
    """
    Builds a custom CNN model based on the provided hyperparameters.
    
    hparams: dict with keys:
       - num_conv_layers: int, number of conv layers.
       - filters: list of int, number of filters for each conv layer.
       - kernel_size: int or list, kernel size (if int, same for all layers).
       - activation: a torch.nn module (e.g., nn.ReLU(), nn.LeakyReLU()).
       - dropout_rate: float, dropout probability.
       - num_classes: int, number of output classes.
       - input_channels: int, number of input channels.
       - image_size: int, width/height of input image (assumed square).
       - use_skip_connections: bool, if True, add a skip connection block.
    """
    num_conv_layers = hparams.get('num_conv_layers', 2)
    filters = hparams.get('filters', [32, 64])
    kernel_size = hparams.get('kernel_size', 3)
    activation = hparams.get('activation', nn.ReLU())
    dropout_rate = hparams.get('dropout_rate', 0.5)
    num_classes = hparams.get('num_classes', 10)
    input_channels = hparams.get('input_channels', 3)
    image_size = hparams.get('image_size', 28)
    use_skip = hparams.get('use_skip_connections', False)
    
    if isinstance(kernel_size, int):
        kernel_sizes = [kernel_size] * num_conv_layers
    else:
        kernel_sizes = kernel_size

    layers = []
    in_channels = input_channels
    for i in range(num_conv_layers):
        out_channels = filters[i] if i < len(filters) else filters[-1]
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_sizes[i], padding=kernel_sizes[i]//2))
        layers.append(activation)
        layers.append(nn.MaxPool2d(2, 2))
        if use_skip and i > 0:
            layers.append(nn.Identity())  # Placeholder for skip connection.
        in_channels = out_channels

    # Compute the spatial size after pooling. Each pooling reduces dimensions by 2.
    factor = 2 ** num_conv_layers
    fc_input_dim = filters[-1] * (image_size // factor) * (image_size // factor)
    
    fc_layers = [
        nn.Linear(fc_input_dim, 128),
        activation,
        nn.Dropout(dropout_rate),
        nn.Linear(128, num_classes)
    ]
    
    model = nn.Sequential(*(layers + fc_layers))
    return model

# -----------------------
# Hyperparameter Definitions for Two Experiments
# -----------------------

hparams1 = {
    'num_conv_layers': 2,
    'filters': [32, 64],
    'kernel_size': 3,
    'activation': nn.ReLU(),
    'dropout_rate': 0.5,
    'num_classes': num_classes,
    'input_channels': 3,
    'image_size': image_size,  # 224
    'use_skip_connections': False
}

hparams2 = {
    'num_conv_layers': 3,
    'filters': [32, 64, 128],
    'kernel_size': 3,
    'activation': nn.LeakyReLU(0.1),
    'dropout_rate': 0.3,
    'num_classes': num_classes,
    'input_channels': 3,
    'image_size': image_size,  # 224
    'use_skip_connections': True
}

model1 = build_custom_cnn(hparams1).to(device)
model2 = build_custom_cnn(hparams2).to(device)

# -----------------------
# Training and Evaluation Functions
# -----------------------

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()  # Ensure labels are 1D
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()
        
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()
        
        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}", flush=True)
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs
    }

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return all_preds, all_labels

def plot_experiments_results(experiments):
    sample_exp = next(iter(experiments.values()))
    num_epochs = len(sample_exp["train_losses"])
    epochs = range(1, num_epochs+1)
    
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    for name, metrics in experiments.items():
        plt.plot(epochs, metrics["train_losses"], label=f"{name} Train Loss")
        plt.plot(epochs, metrics["val_losses"], label=f"{name} Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    
    plt.subplot(1,2,2)
    for name, metrics in experiments.items():
        plt.plot(epochs, metrics["train_accs"], label=f"{name} Train Acc")
        plt.plot(epochs, metrics["val_accs"], label=f"{name} Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# -----------------------
# Train the Two Models
# -----------------------

criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.001)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)
num_epochs_exp = 5

print("Training Experiment 1 - Baseline Model")
metrics1 = train_model(model1, train_loader, val_loader, criterion, optimizer1, num_epochs_exp, device)

print("\nTraining Experiment 2 - Model with LeakyReLU and Skip Connections")
metrics2 = train_model(model2, train_loader, val_loader, criterion, optimizer2, num_epochs_exp, device)

experiments = {
    "Exp1 - Baseline": metrics1,
    "Exp2 - Leaky+Skip": metrics2
}

plot_experiments_results(experiments)

# %% [markdown]
# ###  Final Assignment to this notebook.
# 
# Try to create your network model for the selected dataset, compare your model with transfer learning models, use augmentation, regularization, and optimization strategies to find the best approaching models and check by the visualization techniques whether they work intuitively correct?

# %%



