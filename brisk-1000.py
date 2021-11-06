#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
from gensim.models import Word2Vec
import nltk
stemmer = SnowballStemmer('english')

from numpy import dot
from numpy.linalg import norm


# In[2]:



import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import cv2
import gc


# In[3]:


import os
os.chdir("../")
print("parent path -> ", os.getcwd())


# In[4]:

train_df = pd.read_csv('Project/data/subset/train.csv')
test_df = pd.read_csv('Project/data/subset/test.csv')


# In[5]:


train_image_list = train_df['image'].to_list()


# In[6]:


test_image_list = test_df['image'].to_list()


# # Create 1000 images for training

# In[7]:


train_list = train_image_list
len(train_list)


# # BRISK

# In[17]:


import cv2 
descriptors_list = []
# Initiate BRISK detector
brisk = cv2.BRISK_create(30)

base_path = 'Project/data/train_images/'
image_paths = train_image_list
for image in image_paths:
    image_path = base_path+image
    im = cv2.imread(image_path)
    # Convert the training image to RGB vector
    training_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    # Convert the training image to gray scale vector
    gray_image = cv2.cvtColor(training_image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = brisk.detectAndCompute(gray_image, None)
    descriptors_list.append((image_path, descriptors))
# Extracting descriptor details
descriptors = descriptors_list[0][1]


# In[18]:


descriptors


# In[19]:


ctr = 0
for image_path, descriptor in descriptors_list[1:]:
    ctr = ctr+1
    print("ctr ", ctr)
    if descriptor is not None:
        # Stack arrays in sequence vertically (row wise).
        descriptors = np.vstack((descriptors, descriptor))  

# As kmeans take only float variables
descriptors = descriptors.astype(float)


# # Perform k-means clustering and vector quantization

# In[20]:


from scipy.cluster.vq import kmeans, vq

k = 200  # taking k as 200
code_book, distortion = kmeans(descriptors, k, 1)

image_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    features = descriptors_list[i][1]
    vectors, distance = vq(features,code_book)
    for v in vectors:
        image_features[i][v] += 1

# # scaling

# In[22]:


from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler().fit(image_features)
image_features = standard_scaler.transform(image_features)


# # Create image labels for 1000 images

# In[23]:


y_train = train_df['label_group']
y_arr = y_train.to_numpy()


# In[24]:


my_dict = {}
ptr = -1
modified_labels = []
for label in y_arr:
    if label not in my_dict.keys():
        ptr = ptr+1
        my_dict[label] = ptr
        modified_labels.append(ptr)
    else:
        my_dict[label] = my_dict.get(label)
        modified_labels.append(my_dict.get(label))
image_classes = modified_labels


# # Train an algorithm to discriminate vectors corresponding to positive and negative training images

# In[25]:


image_features


# # Train the Linear SVM

# In[26]:


from sklearn.svm import LinearSVC
clf = LinearSVC(max_iter=50000)  #Default of 100 is not converging
clf.fit(image_features, np.array(image_classes))


# # Train Random forest to compare how it does against SVM

# In[217]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 100, random_state=30)
clf.fit(image_features, np.array(image_classes))


# In[27]:


test_image_paths = test_image_list


# In[30]:


# Create List where all the descriptors will be stored
test_des_list = []

#BRISK
brisk = cv2.BRISK_create(30)
for image in test_image_paths:
    image_path = base_path+image
    im = cv2.imread(image_path)
    # Convert the test image to RGB vector
    test_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    # Convert the test image to gray scale vector
    test_gray = cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = brisk.detectAndCompute(test_gray, None)
    test_des_list.append((image_path, descriptors))

descriptors = test_des_list[0][1]
for image_path, descriptor in test_des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor))

from scipy.cluster.vq import vq    
test_features = np.zeros((len(test_image_paths), k), "float32")
for i in range(len(test_image_paths)):
    vectors, distance = vq(test_des_list[i][1],code_book)
    for v in vectors:
        test_features[i][v] += 1


# # Perform Tf-Idf vectorization

# In[31]:

# # scaling test features

# In[32]:


test_features = standard_scaler.transform(test_features)


# In[33]:


test_features[0].size


# In[34]:


classes_names = test_image_paths

pred = clf.predict(test_features)


# In[37]:


pred


# # Create test labels

# In[38]:


y_test = test_df['label_group']
y_test_arr = y_test.to_numpy()


# In[39]:


y_test_arr


# In[40]:


modified_test_labels = []
for label in y_test_arr:
    if label not in my_dict.keys():
        ptr = ptr+1
        my_dict[label] = ptr
        modified_test_labels.append(ptr)
    else:
        my_dict[label] = my_dict.get(label)
        modified_test_labels.append(my_dict.get(label))


# In[41]:


modified_test_labels


# In[42]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[43]:


accuracy = accuracy_score(modified_test_labels, pred)
print ("accuracy = ", accuracy)