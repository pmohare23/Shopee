#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np
np.random.seed(2018)


# In[2]:

import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[3]:


tf.__version__


# In[4]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[5]:
import os
os.chdir("../")
print("parent path -> ", os.getcwd())

train_df = pd.read_csv('Project/data/subset/train.csv')

train_df


# In[18]:


label_group_list = train_df['label_group'].tolist()
label_group_list


# # Create custom labels on train dataset

# In[19]:


my_dict = {}
ptr = -1
modified_labels = []
for label in label_group_list:
    if label not in my_dict.keys():
        ptr = ptr+1
        my_dict[label] = ptr
        modified_labels.append(ptr)
    else:
        my_dict[label] = my_dict.get(label)
        modified_labels.append(my_dict.get(label))


# In[20]:


modified_labels


# In[21]:


modified_label_group_df = pd.DataFrame(modified_labels, columns = ['modified_label_group'])
modified_label_group_df


# In[22]:


train_df = pd.concat([train_df, modified_label_group_df], axis=1)
train_df


# # Cross validation

# In[23]:


train_df['data'] = train_df['posting_id'].astype(str) + "$" + train_df['image'].astype(str) + "$"         + train_df['image_phash'].astype(str) + "$" + train_df['title'].astype(str) + "$"         + train_df['label_group'].astype(str)


# In[24]:


cols = ['posting_id','image','image_phash','title','label_group']
train_df = train_df.drop(cols, axis=1)


# In[25]:


from sklearn import model_selection

def cross_validation(training_set, split_size = 0.2, random_state = 3, shuffle_state = True):
    """
        Shuffle and split training data depending on given split size

    """
    print(split_size, random_state, shuffle_state)
    return model_selection.train_test_split(training_set.data,
                                                   training_set.modified_label_group,
                                                   test_size=split_size,
                                                   random_state=random_state,
                                                   shuffle=shuffle_state)


# In[26]:


X_train, X_test, y_train, y_test = cross_validation(train_df,
                                                    split_size=0.2,
                                                    random_state=15,
                                                    shuffle_state=True)


# In[27]:


X_train = X_train.str.split('$', expand=True)
X_test = X_test.str.split('$', expand=True)


# In[28]:


y_train.dtypes


# In[29]:


X_train


# In[30]:


X_test = X_test.iloc[:, :-1]
X_test


# In[31]:


X_train.columns = cols
X_test.columns = cols


# In[32]:


X_train.head(10)

# In[40]:


train_image_list = X_train['image'].to_list()
train_image_list


# In[41]:


# y_train = X_train['label_group'].to_list()
y_train = np.asarray(y_train)
y_arr = y_train


# In[42]:


y_arr.shape


# In[43]:


# y_test = X_test['label_group'].to_list()
y_test = np.asarray(y_test)
y_test_arr = y_test


# In[44]:


y_test_arr.shape


# In[45]:


len(y_test_arr)

test_image_list = X_test['image'].to_list()


# In[48]:


len(test_image_list)


# In[49]:


import cv2
res = []
base_path = 'Project/data/train_images/'
for image in train_image_list:
    image_path = base_path+image
    img = cv2.imread(image_path)
    res.append(cv2.resize(img, (224,224)))
    del img


# In[50]:


test_res = []
base_path = 'Project/data/train_images/'
for image in test_image_list:
    image_path = base_path+image
    img = cv2.imread(image_path)
    test_res.append(cv2.resize(img, (224,224)))
    del img
    


# In[82]:


res


# In[51]:


x_arr = np.asarray(res)


# In[52]:


x_arr.shape


# In[53]:


x_test_arr = np.asarray(test_res)
x_test_arr.shape


# In[54]:


y_train


# In[55]:


# y_arr = y_train.to_numpy()
# y_arr.shape
y_arr = y_train


# In[56]:


# y_test_arr = y_test.to_numpy()
# y_test_arr.shape
y_test_arr = y_test


# # Image normalization

# In[57]:


x_arr = x_arr/255


# In[58]:


x_test_arr = x_test_arr/255
x_test_arr


y_arr.size


# In[61]:


len(modified_labels)


# In[62]:


uniqueKeys = set(my_dict.keys())
len(uniqueKeys)


# In[63]:


uniqueValues = set(my_dict.values())
len(uniqueValues)
# In[67]:


modified_label_arr = np.asarray(modified_labels)
modified_label_arr


# In[68]:

# # Take pre-train Tensorflow_hub mobilenet model and retrain it using train images with 375 classes

# In[69]:


feature_extractor_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)


# In[70]:


model = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(len(uniqueKeys))
])

model.summary()


# In[71]:


X_train_scaled = x_arr
y_train_arr = y_arr


# In[76]:


model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

model.fit(X_train_scaled, y_train_arr, epochs=5)


# In[77]:


X_test_scaled = x_test_arr


# In[78]:


model.evaluate(X_test_scaled,y_test_arr)


# In[81]:


y_train_arr


# # Saving the model

# In[84]:


model.save('models/mobilenet_sivyati_model_224.h5')


# In[ ]:




