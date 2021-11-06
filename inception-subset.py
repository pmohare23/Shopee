#!/usr/bin/env python
# coding: utf-8

# In[129]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os


# In[130]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import cv2
import gc


# In[131]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[132]:



# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
#import matplotlib.pyplot as plt


# In[148]:


# re-size all the images to this
IMAGE_SIZE = [224,224]
import os
os.chdir("../")
print("parent path -> ", os.getcwd())
train_path = 'Project/data/folder/train_images_folder'
valid_path = 'Project/data/folder/test_images_folder'


# In[149]:


# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

inception = InceptionV3(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)


# In[150]:


# don't train existing weights
for layer in inception.layers:
    layer.trainable = False


# In[151]:


# useful for getting number of output classes
folders = glob('Project/data/folder/train_images_folder/*')

# In[152]:


# our layers - you can add more if you want
x = Flatten()(inception.output)


# In[153]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=inception.input, outputs=prediction)


# In[154]:

# view the structure of the model
model.summary()


# In[155]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[156]:


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[157]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('Project/data/folder/train_images_folder',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[158]:


test_set = test_datagen.flow_from_directory('Project/data/folder/test_images_folder',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[163]:


test_set


# In[159]:


true_val = test_set.classes
true_val


# In[160]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)


# In[161]:


import matplotlib.pyplot as plt


# In[162]:


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[105]:


# save it as a h5 file
from tensorflow.keras.models import load_model

model.save('Project/models/model_inception.h5')


# In[106]:


y_pred = model.predict(test_set)


# In[108]:


y_pred = np.argmax(y_pred, axis=1)


# # Model Evaluation

from sklearn.metrics import confusion_matrix , classification_report
print("Classification Report: \n", classification_report(true_val, y_pred))




