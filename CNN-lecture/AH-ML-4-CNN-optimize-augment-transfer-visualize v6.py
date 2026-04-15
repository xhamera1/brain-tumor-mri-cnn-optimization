# %% [markdown]
# <table>
# <tr>    
# <td style="text-align: center">
# <h1>Introduction to Comprehensive Training, Augmentation, Transfer Learning and Visualization of CNNs</h1>
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
# # Comprehensive CNN Training, Augmentation, Transfer Learning & Visualization
# 
# This notebook demonstrates:
# - **CNN Training & Optimization:** Building and training a convolutional network from scratch.
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
# Having "few" samples can mean anywhere from a few hundred to few tens of thousands of images. The definition of "few" depends on how many data dimensions we have. We demonstrate it by classifying images of dogs and cats, using a dataset containing 10000 pictures of cats and dogs (5000 cats, 5000 dogs), which input data space dimension is `img_rows` x `img_cols` = 150 x 150 = 22,500. Our workflow will be as follows:
# 
# * First, we will divide this dataset into 6000 training images, 2000 validation images, and 2000 testing images.
# 
# * Next, we will train a new model from scratch, starting with a small convnet on our 6000 training samples, without any regularization, to set a baseline for what can be achieved.
# 
# * Then, we will introduce <b>data augmentation</b>, a powerful technique for mitigating overfitting in computer vision. By leveraging data augmentation, we will improve our network to reach a higher accuracy of the model.
# 
# * Finally, we will apply <b>feature extraction with a pre-trained network</b>, and <b>fine-tuning a pre-trained network</b> to increase the final model peformance.
# 
# These strategies together will constitute your basic future toolbox for tackling the problem of doing computer vision with small datasets. Next, we use <b>transfer learning</b> to train the model faster and better, using pre-trained models on a big dataset to use well-trained features to adapt the model for similar tasks easier and faster!
# 
# For now, let's get started by getting our hands on the data and preparing them for our experiments.
# 
# <img src="http://home.agh.edu.pl/~horzyk/lectures/jupyternotebooks/images/catsvsdogssamples.png" title="examples of cats and dogs" />

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
# ## Downloading the data
# 
# The cats vs. dogs dataset that we will use isn't packaged with Keras, so we have to download and preprocess it. It was made available by <a href="https://www.kaggle.com">Kaggle.com</a> as part of a computer vision competition in late 2013. You can download the original dataset at: <a href="https://www.kaggle.com/c/dogs-vs-cats/data">`https://www.kaggle.com/c/dogs-vs-cats/data`</a> (you will need to create a Kaggle account if you don't already have one). The pictures are medium-resolution (150 x 150 pixels) color JPEGs. 
# 
# Unsurprisingly, the cats vs. dogs Kaggle competition in 2013 was won by entrants who used convnets. The best entries could achieve up to 95% accuracy. In our example, we will get fairly close to this accuracy, even though we will be training our models on 8% of the data that were available to the competitors.
# 
# This original dataset contains 25,000 images of cats and dogs (12,500 from each class) and is 543MB large (when compressed). After downloading and uncompressing it, we will create a new dataset containing three subsets: a training set with 6000 samples of each class, a validation set with 2000 samples of each class, and finally, a test set with 2000 samples of each class. We have to crop the dataset because of the training time. However, if you have enough powerful GPGPU units, you can try to train, validate, and test your final experimental model on the whole dataset.
# 
# Here are a few lines of code to do this:
# 
# This snippet divides the data into training, validation, and test splits. This code creates the necessary directories (if they do not exist) and copies the specified files from the original dataset into the appropriate folders for cats and dogs.

# %%
import os

# Change to your desired directory
os.chdir('/home/ahorzyk/PythonNotebooks/datasets/')   # Change your working directory where your data are stored

# Confirm the change
print("Current working directory:", os.getcwd())

import shutil

# The path to store trained models
models_dir = os.getcwd() + 'models/'   #'C:/ml/models/'
if not os.path.exists(models_dir):
    os.mkdir(models_dir)

# The path to the directory where the original dataset was uncompressed
original_dataset_dir = 'data/train_original'

# The directory where we will store our smaller dataset
base_dir = os.getcwd() + 'data/'
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# Directories for our training, validation and test splits
train_dir = os.path.join(base_dir, 'train')
if not os.path.exists(train_dir):
    os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
if not os.path.exists(validation_dir):
    os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
if not os.path.exists(test_dir):
    os.mkdir(test_dir)

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')
if not os.path.exists(train_cats_dir):
    os.mkdir(train_cats_dir)

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')
if not os.path.exists(train_dogs_dir):
    os.mkdir(train_dogs_dir)

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
if not os.path.exists(validation_cats_dir):
    os.mkdir(validation_cats_dir)

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
if not os.path.exists(validation_dogs_dir):
    os.mkdir(validation_dogs_dir)

# Directory with our test cat pictures
test_cats_dir = os.path.join(test_dir, 'cats')
if not os.path.exists(test_cats_dir):
    os.mkdir(test_cats_dir)

# Directory with our test dog pictures
test_dogs_dir = os.path.join(test_dir, 'dogs')
if not os.path.exists(test_dogs_dir):
    os.mkdir(test_dogs_dir)

# Copy first 1000 cat images to train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(3000)]   # You can use smaller or bigger number or data
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy next 500 cat images to validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(3000, 4000)]   # You can use smaller or bigger number or data
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 cat images to test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(4000, 5000)]   # You can use smaller or bigger number or data
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy first 1000 dog images to train_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(3000)]   # You can use smaller or bigger number or data
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 dog images to validation_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(3000, 4000)]   # You can use smaller or bigger number or data
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
    
# Copy next 500 dog images to test_dogs_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(4000, 5000)]   # You can use smaller or bigger number or data
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)

print("Data has been divided into training, validation, and test sets.")

# Define data directories – update these paths if necessary
train_dir = 'data/train'
validation_dir = 'data/validation'
test_dir = 'data/test'

# Check that the directories exist to avoid FileNotFoundError
assert os.path.exists(train_dir), f"Train directory not found: {train_dir}. Please update the path."
assert os.path.exists(validation_dir), f"Validation directory not found: {validation_dir}. Please update the path."
assert os.path.exists(test_dir), f"Test directory not found: {test_dir}. Please update the path."

# Full paths for train, validation, and test directories
print(f"Train directory is: {os.path.abspath(train_dir)}")
print(f"Validation directory is: {os.path.abspath(validation_dir)}")
print(f"Test directory is: {os.path.abspath(test_dir)}")

# Numbers of train, validation, and test examples
print('Total training cat images:', len(os.listdir(train_cats_dir)))
print('Total training dog images:', len(os.listdir(train_dogs_dir)))
print('Total validation cat images:', len(os.listdir(validation_cats_dir)))
print('Total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('Total test cat images:', len(os.listdir(test_cats_dir)))
print('Total test dog images:', len(os.listdir(test_dogs_dir)))


# %% [markdown]
# So, we have indeed 2000 training images, 1000 validation images, and 1000 test images. In each split, there is the same number of samples from each class: this is <b>a balanced binary classification problem</b>, which means that <b>classification accuracy</b> will be an appropriate measure of success.

# %% [markdown]
# ## 2. Data Preprocessing & Augmentation
# In this section, we assume your training and validation images are organized in directories (e.g., `data/train` and `data/validation`). We use the `ImageDataGenerator` to rescale the images and apply various augmentation techniques (rotation, shifts, shear, zoom, and horizontal flipping) to the training data.
# 
# ## <a href="https://keras.io/api/preprocessing/image/">Data preprocessing</a>
# 
# Data fed to the network must be appropriately formatted into floating point tensors. Currently, our data sits on a drive as JPEG files, so we must perform the following steps to get it into our network:
# 
# * Read the picture files in the JPEG format.
# * Decode the JPEG content to RBG grids of pixels.
# * Convert them into floating point tensors.
# * Rescale the pixel values (between 0 and 255) to the $[0, 1]$ interval because neural networks prefer to deal with small input values. (Note: I pay attention to the coding method $[0, 1]$, which allows you to pay attention only to the strongest features, and not to the lack of them. Then the features should be coded in the range $[-1, 1]$, which together with negative weights can make it possible.)
# 
# Keras has utilities to take care of these steps automatically because it has a module with image processing helper tools, located at `keras.preprocessing.image` (`tensorflow.keras.preprocessing.image`). In particular, it contains the class `ImageDataGenerator`, which allows quickly set up Python generators that can automatically turn image files on disk into batches of pre-processed tensors. We will implement it here.
# 

# %% [markdown]
# ### Option 1 (Categorical)
# 
# - **Model:** Last layer is `Dense(2, activation='softmax')`
# - **Loss:** `categorical_crossentropy`
# - **Generator:** `class_mode='categorical'`
# 
# ### Option 2 (Binary)
# 
# - **Model:** Last layer is `Dense(1, activation='sigmoid')`
# - **Loss:** `binary_crossentropy`
# - **Generator:** `class_mode='binary'`
# 
# Choose the approach that fits your needs and make sure your generator and model/loss function are aligned.
# 
# We will use here the binary representation of the classes (option 2).

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set image size and batch size (adjust as needed)
img_rows, img_cols = 150, 150  # or (224, 224) if you prefer a larger size
img_size = (img_rows, img_cols)
batch_size = 16

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation images
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    #class_mode='categorical'
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=img_size,
    batch_size=batch_size,
    #class_mode='categorical'
    class_mode='binary'
)

# %% [markdown]
# Let's take a look at the output of one of these generators: it yields batches of 150x150 RGB images (shape `(20, 150, 150, 3)`) and binary labels (shape `(20,)`). 20 is the number of samples in each batch (the `batch_size`). Note that the generator yields these batches indefinitely: it just loops endlessly over the images present in the target folder. For this reason, we need to `break` the iteration loop at some point.

# %%
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

# %% [markdown]
# ## 3. Building a simple CNN from Scratch
# 
# Our convnet will be a stack of alternated `Conv2D` (with `relu` activation functions in all hidden layers and `softmax`) and `MaxPooling2D` layers.
# 
# However, since we are dealing with bigger images than for the MNIST data and a more complex problem, we should make our network accordingly larger, i.e., it will have one more `Conv2D` + `MaxPooling2D` stage. This serves both to augment the capacity of the network, and to further reduce the size of the feature maps so that they are not overly large when we reach the `Flatten` layer (not to produce too many parameters in the first `Dense` layer). Here, since we start from inputs of size 150x150 (a somewhat arbitrary choice), we end up with feature maps of size no more than 7x7 right before the `Flatten` layer.
# 
# Note that the depth of the feature maps is progressively increasing in the network (from 32 to 128), while the size of the feature maps is decreasing (from 148x148 to 5x5) as we can see in `model.summary()` below. In this way, we construct most of the convnets.
# 
# Since we are attacking a binary classification problem, we are ending the network with a single unit (a `Dense` layer of size 1) and a `sigmoid` activation. This unit will encode the probability that the network is looking at one class or the other, where one class is represented by 0 and the second class by 1.
# 
# The following code defines a simple convolutional neural network (CNN). You can modify the architecture and hyperparameters as needed.

# %%
#from keras import optimizers
import tensorflow
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_cnn_model(input_shape=(img_rows, img_cols, 3), num_classes=1):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape, name='conv1'),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu', name='conv2'),     # You can use diferent activation functions here
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu', name='conv3'),     # You can use diferent activation functions here
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),     # You can use diferent activation functions here or more dense layers
        #Dropout(0.2),     # If you would like to add regularization
        #Dense(num_classes, activation='softmax')
        Dense(num_classes, activation='sigmoid')
    ])
    return model

# Determine number of classes from the training generator
#num_classes = len(train_generator.class_indices)
#model_cnn = build_cnn_model(input_shape=(img_rows, img_cols, 3), num_classes=num_classes)
#model_cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model_cnn.compile(loss='categorical_crossentropy', optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=1e-4), metrics=['accuracy'])

num_classes = 1
model_cnn = build_cnn_model(input_shape=(img_rows, img_cols, 3), num_classes=num_classes)
#model_cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn.compile(loss='binary_crossentropy', optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=1e-4), metrics=['accuracy'])

model_cnn.summary()


# %% [markdown]
# Let's fit our model to the data using the generator. We do it using the `fit`. It expects as the first argument a Python generator that will yield batches of inputs and targets indefinitely as ours does. Because the data is being generated endlessly, the generator needs to know example how many samples to draw from the generator before declaring an epoch over. This is the role of the `steps_per_epoch` argument: after having drawn `steps_per_epoch` batches from the generator, i.e. after having run for `steps_per_epoch` gradient descent steps, the fitting process will go to the next epoch. In our case, batches are 40-sample large, so it will take 150 batches until we see our target of 6000 samples (40 x 150 = 6000).
# 
# When using `fit`, one may pass a `validation_data`. Importantly, this argument is allowed to be a data generator itself, but it could be a tuple of Numpy arrays as well. If you pass a generator as `validation_data`, then this generator is expected to yield batches of validation data endlessly, and thus you should also specify the `validation_steps` argument, which tells the process how many batches to draw from the validation generator for evaluation. For validation data, we see our target of 1000 samples (40 x 50 = 2000).

# %%
history_cnn = model_cnn.fit(
      train_generator,
      steps_per_epoch=150,
      epochs=40,
      validation_data=validation_generator,
      validation_steps=50)

# %%
# A good practice is to save the trained model for future comparisons or use
model_cnn.save(models_dir + 'cats_and_dogs_small_cnn1.h5')

# Next, you can load the trained model again and use if you need it
#model_cnn.load(models_dir + 'cats_and_dogs_small_cnn1.h5')

# %% [markdown]
# ## 4. Transfer Learning Models & Comparison
# 
# In this section we will:
# 
# - Build custom transfer learning models using several popular pre-trained networks:
#   - **VGG16** (already demonstrated)
#   - **VGG19**
#   - **ResNet50**
#   - **InceptionV3**
#   - **MobileNetV2**
#   - **EfficientNetB0**
#  - Train each model on the same training and validation data.
#  - Compare their training processes (accuracy and loss curves) side-by-side.
# 
# For each model, we:
# 1. Load the base model with ImageNet weights (excluding the top classification layers).
# 2. Freeze the base model layers.
# 3. Add a custom classification head.
# 4. Train the model on the training data and validate on the validation data.
# 
# The trained models are stored in the dictionary `trained_models` for later visualization and evaluation.
# 
#  **Note:** For demonstration, we train for a small number of epochs. In practice, you might use early stopping or train for more epochs.

# %%
import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3, MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, ELU


def build_transfer_model(base_model_fn, input_shape, num_classes):
    """
    Build a transfer learning model using a given base model function.
    The base model is loaded with ImageNet weights, and its layers are frozen.
    A custom head is added on top for classification.
    """
    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False    # False means that we freeze the convolutional base preveting any changes in it

    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.1)(x)     # You can use diferent activation functions here or more dense layers
    x = Dropout(0.15)(x)
    x = Dense(8, activation='elu')(x)     # You can use diferent activation functions here or more dense layers
    x = Dropout(0.2)(x)
    #predictions = Dense(num_classes, activation='softmax')(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Dictionary of models to compare
transfer_models = {
    "VGG16": VGG16,
    "VGG19": VGG19,
    "ResNet50": ResNet50,
    "InceptionV3": InceptionV3,
    "MobileNetV2": MobileNetV2,
    "EfficientNetB0": EfficientNetB0
}

# Dictionaries to store the trained models and their training histories
trained_models = {}
transfer_histories = {}


# %% [markdown]
# ### Training Multiple Transfer Learning Models
# 
# We loop through our selected models, build each one using our custom head, and train on the same data.
# 
# **Important:** Adjust the number of epochs for your needs. For demonstration, we use a low number.

# %%
# Ensure that the number of classes is set based on your training data.
# (Assuming 'train_generator' has been defined previously)
#num_classes = len(train_generator.class_indices)
num_classes = 1

# Number of epochs for transfer learning models (adjust as needed)
epochs_transfer = 40  # For demonstration, consider increasing this for better results
img_rows = 150   # 224
img_cols = 150   # 224
img_size = (img_rows, img_cols)

# Train each model and store its history
for model_name, base_model_fn in transfer_models.items():
    print(f"Training {model_name} model...")
    model = build_transfer_model(base_model_fn, input_shape=(img_rows, img_cols, 3), num_classes=num_classes)
    history = model.fit(
        train_generator,
        epochs=epochs_transfer,
        validation_data=validation_generator,
        verbose=1
    )
    trained_models[model_name] = model
    transfer_histories[model_name] = history.history
    print(f"Finished training {model_name}.\n")


# %% [markdown]
# ### Comparing Training Processes
# 
# Now we plot the training and validation accuracies and losses for each transfer learning model.
# 

# %%
# Plot training accuracy for all models
plt.figure(figsize=(12, 6))
for model_name, hist in transfer_histories.items():
    plt.plot(hist['accuracy'], label=f'{model_name} Train Acc')
plt.title("Training Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Plot validation accuracy for all models
plt.figure(figsize=(12, 6))
for model_name, hist in transfer_histories.items():
    plt.plot(hist['val_accuracy'], label=f'{model_name} Val Acc')
plt.title("Validation Accuracy Comparison")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# %%
# Plot training loss for all models
plt.figure(figsize=(12, 6))
for model_name, hist in transfer_histories.items():
    plt.plot(hist['loss'], label=f'{model_name} Train Loss')
plt.title("Training Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot validation loss for all models
plt.figure(figsize=(12, 6))
for model_name, hist in transfer_histories.items():
    plt.plot(hist['val_loss'], label=f'{model_name} Val Loss')
plt.title("Validation Loss Comparison")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


# %% [markdown]
# ## Summary
# 
# In this extended section, we:
# 
# - Built transfer learning models using VGG16, VGG19, ResNet50, InceptionV3, MobileNetV2, and EfficientNetB0.
# - Trained each model on the same dataset.
# - Compared their training and validation performance using accuracy and loss plots.
# 
# You can now analyze these plots to determine which model performs best on your data. Adjust training epochs, learning rates, or unfreeze additional layers for fine-tuning as needed.
# 

# %% [markdown]
# ## 5. Visualization Techniques and Visualizing Internal Representations Across Networks
# 
# In this section, we apply several visualization techniques to compare the internal quality of the trained filters and outputs from different transfer learning networks:
# - **Confusion Matrices:** To compare the classification performance on the validation set.
# - **Filter Visualization:** We display the weights of the first convolutional layer.
# - **Activation Maps:** We show the output feature maps from the first convolutional layer for a sample image.
# - **Grad-CAM Heatmaps:** (See previous sections) Highlighting class-discriminative regions.
# 
# These visualizations help you understand not only the performance but also the internal representations learned by each model.

# %% [markdown]
# ### Loading the Sample Image for comparisons
# 
# We load a sample image (e.g., located at `path/sample.jpg`). Adjust the path as necessary. The image is preprocessed for input into the models, and we also keep a copy of the original image for overlaying the heatmap.

# %%
def load_and_preprocess_image(img_path, target_size=(img_rows, img_cols)):
    """
    Loads an image from disk, resizes it, and scales pixel values to [0,1].
    Returns both the preprocessed image for model input and the original image as a NumPy array.
    """
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # Also return the original image in its array form (scaled [0,1])
    return img_array, np.array(img)

# %%
sample_img_path = os.getcwd() + '/data/test/cats/cat.1500.jpg'  # Update this path to your sample image
img_array, original_img = load_and_preprocess_image(sample_img_path, target_size=(img_rows, img_cols))

# %% [markdown]
# ### 5.1. Plotting Confusion Matrices

# %%
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(model, model_name, validation_generator, num_classes):
    """
    Computes predictions on the validation set and plots the confusion matrix.
    For binary classification (num_classes == 1), uses a threshold of 0.5.
    For multiclass classification (num_classes > 1), uses argmax over predictions.
    
    Parameters:
      model: the trained model.
      model_name: name of the model (for labeling).
      validation_generator: the validation data generator.
      num_classes: number of output classes (1 for binary with sigmoid, >1 for multiclass with softmax).
    """
    # Reset generator and determine steps.
    validation_generator.reset()
    val_steps = validation_generator.samples // validation_generator.batch_size + 1
    predictions = model.predict(validation_generator, steps=val_steps, verbose=0)
    
    if num_classes == 1:
        # Binary classification: predictions is a one-element per sample
        predicted_classes = (predictions > 0.5).astype("int32").flatten()
    else:
        # Multiclass classification: predictions is an array with shape (samples, num_classes)
        predicted_classes = np.argmax(predictions, axis=1)
    
    true_classes = validation_generator.classes
    class_labels = list(validation_generator.class_indices.keys())
    
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


# %%
# Example: Loop over each model to plot its confusion matrix.
num_classes = 1
for model_name, model in trained_models.items():
    plot_confusion_matrix(model, model_name, validation_generator, num_classes=num_classes)

# %% [markdown]
# Based on the accuracies and confusion matrices, we can see, that the models require longer training process than 30 epochs and, next, probably fine-tuning of the transfered models.

# %% [markdown]
# ### 5.2. Visualizing Filters of the First Convolutional Layer

# %%
import matplotlib.pyplot as plt
import tensorflow as tf

def visualize_filters(model, model_name, n_filters=6):
    """
    Visualizes the filters of the first convolutional layer in the model.
    """
    first_conv_layer_name = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            first_conv_layer_name = layer.name
            break
    if first_conv_layer_name is None:
        print(f"No Conv2D layer found for {model_name}.")
        return

    layer = model.get_layer(first_conv_layer_name)
    weights = layer.get_weights()
    if len(weights) == 2:
        filters, biases = weights
    elif len(weights) == 1:
        filters = weights[0]
    else:
        print(f"Unexpected number of weight arrays for layer {first_conv_layer_name}.")
        return

    # Normalize filter values for visualization.
    n_filters = min(n_filters, filters.shape[-1])
    
    fig, axes = plt.subplots(1, n_filters, figsize=(20, 8))
    for i in range(n_filters):
        f = filters[:, :, :, i]
        # Normalize to 0-1 for display.
        f_min, f_max = f.min(), f.max()
        f = (f - f_min) / (f_max - f_min + 1e-8)
        # If the filter has more than 3 channels, display the first three channels.
        if f.shape[-1] >= 3:
            ax_img = f[:, :, :3]
        else:
            ax_img = f[:, :, 0]
        axes[i].imshow(ax_img)
        axes[i].axis('off')
        axes[i].set_title(f"{model_name}\n{first_conv_layer_name}\nFilter {i+1}")
    plt.show()


# %%
# Check if the trained_models dictionary exists to perform visualization
if 'trained_models' not in globals():
    print("Error: 'trained_models' dictionary not found. Please run the training cells first to create and store your models.")
else:
    # Example: Loop over all trained models to visualize their first conv filters.
    for model_name, model in trained_models.items():
        visualize_filters(model, model_name)

# %% [markdown]
# ### 5.3. Visualizing Activation Maps from the First Convolutional Layer

# %%
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def display_activation_maps(model, sample_img, layer_names):
    """
    Displays activation maps for the given layers using the sample image.
    """
    # Preprocess image: assume sample_img is a file path.
    img = load_img(sample_img, target_size=(img_rows, img_cols))
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0  # Scale pixel values
    
    # Create a model that outputs the activations for the selected layers.
    outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=outputs)
    activations = activation_model.predict(img_tensor)
    
    for idx, activation in enumerate(activations):
        print(f"Activation shape for layer {layer_names[idx]}: {activation.shape}")
        num_features = activation.shape[-1]
        cols = int(np.sqrt(num_features))
        rows = int(np.ceil(num_features / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
        fig.suptitle(f"Activations from {layer_names[idx]}")
        for i in range(num_features):
            ax = axes[i // cols, i % cols]
            # If activation has batch dimension (4D), use activation[0, :, :, i]; 
            # if not (3D), then use activation[:, :, i]
            if activation.ndim == 4:
                ax.imshow(activation[0, :, :, i], cmap='viridis')
            elif activation.ndim == 3:
                ax.imshow(activation[:, :, i], cmap='viridis')
            ax.axis('off')
        plt.tight_layout()
        plt.show()

def display_first_conv_activation(model, sample_img, model_name):
    """
    Finds the first Conv2D layer and displays its activation maps.
    """
    first_conv_layer_name = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            first_conv_layer_name = layer.name
            break
    if first_conv_layer_name is None:
        print(f"No Conv2D layer found for {model_name}.")
        return
    print(f"Displaying activations for {model_name} - {first_conv_layer_name}")
    display_activation_maps(model, sample_img, [first_conv_layer_name])


# %%
# Check if the trained_models dictionary exists to perform visualization
if 'trained_models' not in globals():
    print("Error: 'trained_models' dictionary not found. Please run the training cells first to create and store your models.")
else:
    # Loop over models and display their first conv layer activations.
    for model_name, model in trained_models.items():
        display_first_conv_activation(model, sample_img_path, model_name)

# %% [markdown]
# ### 5.4. Visualizing and Comparing Activation Maps Using Grad-CAM Heatmaps
# 
# In this section, we compute Grad-CAM heatmaps for a sample image across different transfer learning models. Each model uses its recommended last convolutional layer to generate a heatmap overlay on the original image. The recommended target layers are:
# 
# - **VGG16:** `block5_conv3`
# - **VGG19:** `block5_conv4`
# - **ResNet50:** `conv5_block3_out`
# - **InceptionV3:** `mixed10`
# - **MobileNetV2:** `Conv_1`
# - **EfficientNetB0:** `top_conv`
# 
# The resulting grid of images lets you visually compare the activation maps produced by each network.

# %%
!pip install opencv-python

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def generate_gradcam(model, img_array, target_layer_name, pred_index=None):
    """
    Computes the Grad-CAM heatmap for a given model and image.
    """
    # Create a model that outputs both the target layer and final predictions
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    # Compute the gradient of the target class with respect to the feature maps
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the feature maps by the pooled gradients and sum
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Apply ReLU and normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap_on_image(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):  # cv2.COLORMAP_HOT
    """
    Overlays a heatmap on the original image.
    
    Parameters:
      img: the original image (assumed scaled 0-1 or 0-255)
      heatmap: the raw heatmap output from Grad-CAM
      alpha: blending factor for overlay
      colormap: OpenCV colormap to apply
    """
    # Resize the heatmap to match the image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # If the image is not in uint8, convert it
    if img.dtype != np.uint8:
        img = np.uint8(255 * img)
    
    # Combine the heatmap with the original image
    superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img

# %% [markdown]
# ### Mapping of Models to Target Layers
# 
# Below is a dictionary mapping each transfer learning model to the layer used for computing Grad-CAM:
# - **VGG16:** `block5_conv3`
# - **VGG19:** `block5_conv4`
# - **ResNet50:** `conv5_block3_out`
# - **InceptionV3:** `mixed10`
# - **MobileNetV2:** `Conv_1`
# - **EfficientNetB0:** `top_conv`
# 

# %%
# You can define also some other target layers:
target_layers = {
    "VGG16": "block5_conv3",
    "VGG19": "block5_conv4",
    "ResNet50": "conv5_block3_out",
    "InceptionV3": "mixed10",
    "MobileNetV2": "Conv_1",
    "EfficientNetB0": "top_conv"
}

# %% [markdown]
# ## Generating and Displaying Grad-CAM Heatmaps
# 
# We loop through each trained model, compute the Grad-CAM heatmap using the specified target layer, overlay the heatmap on the original image, and display the results in a grid format. This lets you compare how each network “sees” the same image.
# 

# %%
# Check if the trained_models dictionary exists to perform visualization
if 'trained_models' not in globals():
    print("Error: 'trained_models' dictionary not found. Please run the training cells first to create and store your models.")
else:
    num_models = len(trained_models)
    plt.figure(figsize=(15, 10))
    
    for i, (model_name, model) in enumerate(trained_models.items()):
        # Get the target layer for the current model
        target_layer = target_layers.get(model_name, None)
        if target_layer is None:
            print(f"No target layer specified for model {model_name}. Skipping.")
            continue
        
        # Compute the Grad-CAM heatmap
        heatmap = generate_gradcam(model, img_array, target_layer)
        
        # Overlay the heatmap on the original image
        overlay_img = overlay_heatmap_on_image(original_img, heatmap)
        
        # Plot the result
        plt.subplot(2, 3, i + 1)
        # Convert color from BGR to RGB for proper display
        overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
        plt.imshow(overlay_rgb)
        plt.title(model_name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Summary
# 
# In this section we compared the trained networks using several visualization techniques:
# - **Confusion Matrices:** Comparing classification performance across networks.
# - **Filter Visualization:** Displaying the filters of the first convolutional layer reveals what low-level features each network is learning.
# - **Activation Maps:** Examining the output of the first convolutional layer for a sample image helps compare how each network processes the input.
# - **Grad-CAM Heatmaps:** (Previously implemented) Highlight class-discriminative regions.
# 
# In this section, we also:
# - Defined helper functions to compute Grad-CAM heatmaps and overlay them on input images.
# - Mapped each transfer learning model to its recommended last convolutional layer.
# - Loaded a sample image and computed activation maps (as heatmap overlays) for each model.
# 
# This setup allows you to visually compare how different networks perceive the same image. Adjust the target layers, blending parameters, or colormap as needed.
# 
# By analyzing these visualizations side by side, you can assess not only the performance (via confusion matrices) but also the internal quality of the learned filters and activations. Adjust layer selections, sample images, and visualization parameters as needed for deeper insights.

# %% [markdown]
# ## 6. Conclusion & Further Work
# 
# In this notebook, we have:
# 
# - Built and trained a CNN from scratch.
# - Enhanced our dataset using data augmentation.
# - Leveraged transfer learning with a pre-trained VGG16 network.
# - Visualized training performance and intermediate CNN representations.
# 
# **Future Directions:**
# 
# - Experiment with alternative architectures or deeper networks.
# - Refine augmentation strategies and hyperparameters (e.g. activation functions).
# - Use different sizes of training data.
# - Implement advanced visualization methods such as Grad-CAM for improved interpretability.
# - Gradually unfreeze last layers or blocks of the convolutional bases for further tuning.
# - Choose a different dataset for training and validation.
# - If you use more classes than two, use cathegorical-crossentropy instead of binary-crossentrophy and soft-max instead of sigmoid activation function in the last layer.
# 
# This integrated approach provides a solid framework for developing and analyzing CNNs.
# 

# %% [markdown]
# ## Assignment 1
# 
# Unfreezing some of the later layers of your convolutional base is a common fine-tuning strategy that can lead to improved performance on your specific dataset. The typical workflow is:
# 
# ### Initial Training:
# Freeze the convolutional base and train only the new classification head. This allows the new layers to learn to interpret the pre-trained features.
# 
# ### Fine-Tuning:
# Unfreeze some of the last layers of the base model (e.g., the top few convolutional blocks) and continue training—but usually with a lower learning rate. This lets the pre-trained weights adjust more subtly to your new dataset without overfitting.
# 
# ### Points to Consider:
# 1. Choosing Layers to Unfreeze:
# - Experiment with how many layers you unfreeze. Sometimes unfreezing just the top block of convolutional layers is sufficient.
# 
# 2. Learning Rate:
# - Use a lower learning rate (like 1e-5 or 1e-6) during fine-tuning so that the adjustments to the pre-trained weights are more gradual.
# 
# 3. Monitor Overfitting:
# - Fine-tuning can sometimes lead to overfitting if your dataset is small. Keep an eye on the validation loss and consider using early stopping or regularization if necessary.
# 
# This approach should allow your model to adjust its learned features more precisely to your specific task, potentially leading to better results.
# 
# Below is an example code snippet to unfreeze the last few layers for fine-tuning:

# %% [markdown]
# ## Fine-tuning of the tranferred models
# 
# Below is an example that demonstrates two strategies for improving performance through fine-tuning:
# 
# **Gradual Unfreezing:**  
# Start with the pre-trained base frozen, then (after some initial training) unfreeze a block of layers from the base model and recompile with a lower global learning rate.
# 
# **Layer-Specific Learning Rates:**  
# Use a custom optimizer wrapper (e.g. using a library like `keras_lr_multiplier`
# 
# In this code:
# 
# - We loop over each transfer learning model, build it with a frozen base, and perform initial training.
# 
# - Depending on the flag (use_layer_specific_lr), we either apply layer-specific learning rates using the keras_lr_multiplier package or gradually unfreeze the last few layers and recompile with a lower learning rate.
# 
# - Finally, we compute and plot the confusion matrix for each trained model on the validation set.
# 
# Adjust the parameters (epochs, learning rates, number of layers to unfreeze) as needed for your dataset and hardware.
# 

# %%
import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3, MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Build transfer learning model function for binary classification.
def build_transfer_model(base_model_fn, input_shape, num_classes=1):
    """
    Builds a transfer learning model:
      - Loads the pre-trained base with ImageNet weights (without the top).
      - Initially freezes all base layers.
      - Adds a custom head.
      - Uses a Dense layer followed by LeakyReLU.
      - For binary classification, final Dense layer has 1 unit with sigmoid activation.
    """
    base_model = base_model_fn(weights='imagenet', include_top=False, input_shape=input_shape)
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    # Store the base model so we can refer to it later for fine-tuning.
    model.base_model = base_model

    model.compile(optimizer=Adam(learning_rate=1e-4), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    return model

# Define initial parameters.
img_rows, img_cols = 150, 150
input_shape = (img_rows, img_cols, 3)
initial_epochs = 10
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs

# Conditional flag: set to True to use layer-specific learning rates.
use_layer_specific_lr = False

# Define the dictionary of transfer learning base models.
transfer_models = {
    "VGG16": VGG16,
    "VGG19": VGG19,
    "ResNet50": ResNet50,
    "InceptionV3": InceptionV3,
    "MobileNetV2": MobileNetV2,
    "EfficientNetB0": EfficientNetB0
}

trained_models = {}
transfer_histories = {}

# Loop over each model.
for model_name, base_model_fn in transfer_models.items():
    print(f"Training {model_name} model...")
    model = build_transfer_model(base_model_fn, input_shape, num_classes=1)
    
    # Initial training phase (with the base model frozen).
    history_initial = model.fit(
        train_generator,           # Assume train_generator is defined
        epochs=initial_epochs,
        validation_data=validation_generator,  # Assume validation_generator is defined
        verbose=1
    )
    
    # Fine-tuning phase.
    if use_layer_specific_lr:
        # Option 2: Layer-Specific Learning Rates.
        # Ensure keras_lr_multiplier is installed: pip install keras_lr_multiplier
        from keras_lr_multiplier import LRMultiplier
        
        multipliers = {}
        for layer in model.layers:
            if 'block' in layer.name or 'conv' in layer.name:
                multipliers[layer.name] = 0.1  # Lower LR for pre-trained layers.
            else:
                multipliers[layer.name] = 1.0  # Full LR for new head layers.
        
        base_optimizer = Adam(learning_rate=1e-5)
        optimizer = LRMultiplier(base_optimizer, multipliers=multipliers)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        print(f"{model_name}: Using layer-specific learning rates.")
    else:
        # Option 1: Gradual Unfreezing.
        # Use the stored base_model attribute instead of model.layers[0].
        for layer in model.base_model.layers[-4:]:
            layer.trainable = True
        model.compile(optimizer=Adam(learning_rate=1e-5), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        print(f"{model_name}: Using gradual unfreezing with a lower learning rate.")
    
    history_finetune = model.fit(
        train_generator,
        epochs=total_epochs,
        initial_epoch=initial_epochs,
        validation_data=validation_generator,
        verbose=1
    )
    
    trained_models[model_name] = model
    transfer_histories[model_name] = {"initial": history_initial.history, "fine_tune": history_finetune.history}
    print(f"Finished training {model_name}.\n")

# Compare confusion matrices for all trained models.
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(model, model_name, validation_generator, num_classes):
    """
    Computes predictions on the validation set and plots the confusion matrix.
    For binary classification (num_classes == 1), uses thresholding at 0.5.
    """
    validation_generator.reset()
    val_steps = validation_generator.samples // validation_generator.batch_size + 1
    predictions = model.predict(validation_generator, steps=val_steps, verbose=0)
    
    if num_classes == 1:
        predicted_classes = (predictions > 0.5).astype("int32").flatten()
    else:
        predicted_classes = np.argmax(predictions, axis=1)
    
    true_classes = validation_generator.classes
    class_labels = list(validation_generator.class_indices.keys())
    
    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"Confusion Matrix for {model_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

for model_name, model in trained_models.items():
    plot_confusion_matrix(model, model_name, validation_generator, num_classes=1)


# %%



