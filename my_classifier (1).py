#!/usr/bin/env python
# coding: utf-8

# ## Prepare the workspace

# In[1]:


# Before you proceed, update the PATH
import os
os.environ['PATH'] = f"{os.environ['PATH']}:/root/.local/bin"
os.environ['PATH'] = f"{os.environ['PATH']}:/opt/conda/lib/python3.6/site-packages"
# Restart the Kernel at this point. 


# In[2]:


# Do not execute the commands below unless you have restart the Kernel after updating the PATH. 
get_ipython().system('python -m pip install torch==1.0.0')


# In[3]:


# Check torch version and CUDA status if GPU is enabled.
import torch
print(torch.__version__)
print(torch.cuda.is_available()) # Should return True when GPU is enabled. 


# # Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[4]:


# Imports here
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn # building blocks for creating neural network models
import torch.optim as optim # provides optimization algorithms like SGD (Stochastic Gradient Descent)
import torchvision # provides popular datasets, model architectures, and image transformations
from torchvision import datasets, transforms # common image transformations and data preprocessing operations
from torchvision.datasets import ImageFolder # loads images with folder structure
from torch.utils.data import DataLoader # create batches of data for training and evaluation
import scipy.io # scipy.io is a module for working with MATLAB files
from PIL import Image # Python Imaging Library (PIL) for working with images
import urllib.request # urllib.request is used for downloading data from URLs
import tarfile # tarfile is a module for working with tar archives
from torchvision.datasets.utils import download_url 


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). 

# If you do not find the `flowers/` dataset in the current directory, **/workspace/home/aipnd-project/**, you can download it using the following commands. 
# 
# ```bash
# !wget 'https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz'
# !unlink flowers
# !mkdir flowers && tar -xzf flower_data.tar.gz -C flowers
# ```
# 

# ## Data Description
# The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[5]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[6]:


# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'valid': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

# TODO: Load the datasets with ImageFolder
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
}
# TODO: Using the image datasets and the trainforms, define the dataloaders
batch_size = 32
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size),
}

# Print the number of images in each dataset
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
print("Dataset Sizes:", dataset_sizes)


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[7]:


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

#We have 102 categories
class_label = 102
#Lets see the classes
class_name = cat_to_name[str(class_label)]
print(f"Predicted class name: {class_name}")


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.
# 
# One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to
# GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
# 
# ## Note for Workspace users: 
# If your network is over 1 GB when saved as a checkpoint, there might be issues with saving backups in your workspace. Typically this happens with wide dense layers after the convolutional layers. If your saved checkpoint is larger than 1 GB (you can open a terminal and check with `ls -lh`), you should reduce the size of your hidden layers and train again.

# In[8]:


# TODO: Build and train your network
# i will build my custom neural network architecture
class CustomClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CustomClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 28 * 28, 512)  
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)  
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Initializing my custom classifier
num_classes = 102  # Number of classes 
custom_classifier = CustomClassifier(num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(custom_classifier.parameters(), lr=0.001)


# In[9]:


# Move the custom classifier to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
custom_classifier.to(device)


# In[10]:


# Training 
num_epochs = 10

for epoch in range(num_epochs):
    custom_classifier.train()
    running_loss = 0.0

    for inputs, labels in dataloaders['train']:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = custom_classifier(inputs)
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(dataloaders['train'])
    print(f'Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f}')

print("Training complete!")

# Saving myclassifier model
torch.save(custom_classifier.state_dict(), 'custom_classifier.pth')


# In[ ]:





# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[11]:


# TODO: Do validation on the test set
# Evaluation function for validation and test
def evaluate_model(model, dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Accumulate loss
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(dataloader)
    accuracy = (correct_predictions / total_samples) * 100.0

    return avg_loss, accuracy

# Evaluate on validation dataset
val_loss, val_accuracy = evaluate_model(custom_classifier, dataloaders['valid'], criterion)
print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {val_accuracy:.2f}%')

# Evaluate on test dataset
test_loss, test_accuracy = evaluate_model(custom_classifier, dataloaders['test'], criterion)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.2f}%')


# In[12]:


import json

# Load the JSON file that maps class labels to class names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Define a list of image file paths for prediction
image_paths_to_predict = 'https://jk3d44a3av.prod.udacity-student-workspaces.com/edit/home/aipnd-project/flower_data.tar.gz'
# # Iterate through the list of image file paths
# for image_path in image_paths_to_predict:
#     # Load and preprocess the image
#     processed_image = process_image(image_path)

#     # Predict the top 5 classes
#     top_probs, top_classes = predict(image_path, model, topk=5)

#     # Get class names for the top predicted classes
#     class_names = [cat_to_name[class_label] for class_label in top_classes]

#     # Display the image and top predicted classes
#     show_image_with_predictions(image_path, model)

#     # Print the top predicted classes and their probabilities
#     print(f"Image Path: {image_path}")
#     for i in range(len(class_names)):
#         print(f"Class: {class_names[i]}, Probability: {top_probs[i]:.4f}")
#     print("\n")


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].class_to_idx`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[13]:


# TODO: Save the checkpoint 
# Save my clasiffier
torch.save(custom_classifier.state_dict(), 'my_classifier.pth')

print("Custom classifier model saved.")


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[14]:


# TODO: Write a function that loads a checkpoint and rebuilds the model

def load_model(filepath, num_classes):
    # Initializing instance for my classifier
    model = CustomClassifier(num_classes)

    # Load the saved model state dictionary
    model.load_state_dict(torch.load(filepath))

   
    model.eval()

    return model


num_classes = 102  
saved_model_path = 'my_classifier.pth'  

# Load the saved model
loaded_model = load_model(saved_model_path, num_classes)




# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[15]:


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    preprocess = transforms.Compose([
        transforms.Resize(256),             # Resize the image to 256x256 pixels
        transforms.CenterCrop(224),         # Crop the center 224x224 portion of the image
        transforms.ToTensor(),              # Convert to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])

    # Open the image using PIL
    image = Image.open(image_path)
    
    # Apply the transformations to the image
    img_tensor = preprocess(image)
    
    # Convert the tensor to a NumPy array
    img_np = np.array(img_tensor)
    
    return img_np


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[16]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[17]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    processed_image = process_image(image_path)
    
    # Convert the NumPy array to a PyTorch tensor
    image_tensor = torch.tensor(processed_image)
    
    # Add a batch dimension to the image tensor
    image_tensor = image_tensor.unsqueeze(0)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Move the image tensor to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_tensor = image_tensor.to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)
    
    # Calculate class probabilities and retrieve the top K classes and their indices
    probabilities = torch.exp(output)
    top_probabilities, top_indices = torch.topk(probabilities, topk)
    
    # Convert top_indices to class labels using a mapping (e.g., class_to_idx)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]
    
    # Convert tensors to NumPy arrays for easier handling
    top_probabilities = top_probabilities[0].cpu().numpy()
    
    return top_probabilities, top_classes


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[18]:


# Load the JSON file that maps class labels to class names
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Function to preprocess an image
def process_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)

# Function to predict the top-k classes for an image
def predict(image_path, model, topk=5):
    # Load and preprocess the image
    image = Image.open(image_path)
    image = process_image(image)
    
    # Add a batch dimension and move to the device
    image = image.unsqueeze(0).to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Make the prediction
    with torch.no_grad():
        output = model(image)
    
    # Calculate probabilities and class indices
    probabilities = torch.softmax(output, dim=1)
    top_probabilities, top_indices = torch.topk(probabilities, topk)
    
    # Convert indices to class labels
    top_classes = [list(model.class_to_idx.keys())[list(model.class_to_idx.values()).index(idx)]
                   for idx in top_indices.cpu().numpy()[0]]
    
    return top_probabilities.cpu().numpy()[0], top_classes

# Function to display an image with the top-k predicted classes
def show_image_with_predictions(image_path, model, topk=5):
    # Predict the top-k classes
    top_probs, top_classes = predict(image_path, model, topk=topk)
    
    # Get class names for the top predicted classes
    class_names = [cat_to_name[class_label] for class_label in top_classes]
    
    # Load and display the image
    img = Image.open(image_path)
    plt.figure(figsize=(10, 5))
    
    # Subplot for the image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    
    # Subplot for the top-k predicted classes and their probabilities
    plt.subplot(1, 2, 2)
    y_pos = range(len(class_names))
    plt.barh(y_pos, top_probs, align='center')
    plt.yticks(y_pos, class_names)
    plt.gca().invert_yaxis()
    plt.xlabel('Probability')
    
    plt.tight_layout()
    plt.show()

# Usage example
model = load_model('my_classifier.pth', num_classes=102)  # Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set the device (GPU or CPU)

for image_path in image_paths_to_predict:
    show_image_with_predictions(image_path, model)


# In[ ]:





# ## Reminder for Workspace users
# If your network becomes very large when saved as a checkpoint, there might be issues with saving backups in your workspace. You should reduce the size of your hidden layers and train again. 
#     
# We strongly encourage you to delete these large interim files and directories before navigating to another page or closing the browser tab.

# In[ ]:


# TODO remove .pth files or move it to a temporary `~/opt` directory in this Workspace

import shutil

# Define the path to the temporary directory
opt_directory = '~/opt'  # Replace with the desired directory path

# Create the temporary directory if it doesn't exist
os.makedirs(opt_directory, exist_ok=True)

# List all files in the current directory
files = os.listdir()

# Iterate through the files and move or remove .pth files
for file in files:
    if file.endswith(".pth"):
        file_path = os.path.join(os.getcwd(), file)
        if os.path.exists(file_path):
            # Move the .pth file to the temporary directory
            shutil.move(file_path, os.path.join(opt_directory, file))
            print(f"Moved {file} to {opt_directory}")
        else:
            print(f"File {file} does not exist in the current directory.")


# In[ ]:




