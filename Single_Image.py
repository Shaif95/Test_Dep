#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import tensorflow as tf
from PIL import Image
import numpy as np

# Define the folder location
folder_location = "D:\Images_001"

# List all folders in the given location
folders = os.listdir(folder_location)

# Define the desired image size
target_size = (40, 40)

# Initialize an empty list to store the processed images
processed_images = []

# Iterate through each folder
for folder in folders:
    # Construct the folder path
    folder_path = os.path.join(folder_location, folder)
    
    # List all images in the folder
    images = os.listdir(folder_path)
    
    # Iterate through each image
    for image_name in images:
        # Construct the image path
        image_path = os.path.join(folder_path, image_name)
        
        # Open the image using PIL
        image = Image.open(image_path)
        
        # Resize the image using TensorFlow
        resized_image = tf.image.resize_with_crop_or_pad(
            tf.keras.preprocessing.image.img_to_array(image),
            target_size[0],
            target_size[1]
        )
        
        # Normalize the data for deep learning
        normalized_image = (resized_image - 127.5) / 127.5
        
        # Append the processed image to the list
        processed_images.append(normalized_image)

# Convert the list of images to a NumPy array
processed_images = np.array(processed_images)
processed_images.shape


# In[ ]:





# In[7]:


from tensorflow.keras.models import load_model

# Define the path to the model file
model_path = r'D:\new_model\my_model.h5'

# Define the custom metric function
def get_f1(y_true, y_pred):
    return 0  # Blank function

# Load the model with custom_objects argument
model = load_model(model_path, custom_objects={'get_f1': get_f1})

# Print the model summary
model.summary()


# In[8]:


x = model.predict(processed_images)
y = np.round(x)
res = []
for i in y:
    if(i[0]==1):
        res.append(0)
    elif(i[1]==1):
        res.append(1)
    else:
        res.append(2)


# In[8]:





# In[9]:


size = len(res)
my_list = [0] * size

# Calculate accuracy
total = len(my_list)
correct = sum(1 for x, y in zip(my_list, res) if x == y)
accuracy = correct / total

print("Accuracy:", accuracy)


# In[22]:





# In[ ]:





# In[24]:





# In[25]:





# In[ ]:





# In[ ]:





# In[ ]:




