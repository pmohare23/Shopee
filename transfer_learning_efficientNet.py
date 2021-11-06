#!/usr/bin/env python
# coding: utf-8

# In[133]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np

# In[134]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[135]:


tf.__version__


# In[136]:


import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# In[137]:


import os
os.chdir("../")
print("parent path -> ", os.getcwd())


# In[138]:


train_df = pd.read_csv('Project/data/subset/train.csv')


# In[139]:


valid_df = pd.read_csv('Project/data/labels_375.csv')


# In[140]:


valid_df = valid_df.rename(columns={"LabelGroup": "label_group"})
valid_df


# In[141]:


res = pd.merge(train_df, valid_df, how='left', on=['label_group'])
filtered_df = res[res['count'].notnull()]
train_df = filtered_df.drop(columns=['count'])
train_df


# In[142]:


existing_test_df = pd.read_csv('Project/data/test_1121.csv')
existing_test_df


# In[143]:


res_test = pd.merge(existing_test_df, valid_df, how='left', on=['label_group'])
filtered_test_df = res_test[res_test['count'].notnull()]


# In[144]:


filtered_test_df.count()


# In[145]:


n = len(pd.unique(filtered_test_df['label_group']))
n

common_df = train_df.merge(filtered_test_df, on=['posting_id'])
minus_df = train_df[~train_df.posting_id.isin(common_df.posting_id)]
minus_df                   


# In[147]:


minus_df.count()

train_df = minus_df

train_df


# In[150]:


train_df.columns


# In[151]:


train_df['data'] = train_df['posting_id'].astype(str) + "$" + train_df['image'].astype(str) + "$"         + train_df['image_phash'].astype(str) + "$" + train_df['title'].astype(str)


# In[152]:


cols = ['posting_id','image','image_phash','title']
train_df = train_df.drop(cols, axis=1)


# In[153]:


from sklearn import model_selection

def cross_validation(training_set, split_size = 0.2, random_state = 3, shuffle_state = True):
    """
        Shuffle and split training data depending on given split size

    """
    print(split_size, random_state, shuffle_state)
    return model_selection.train_test_split(training_set.data,
                                                   training_set.label_group,
                                                   test_size=split_size,
                                                   random_state=random_state,
                                                   shuffle=shuffle_state)


# In[154]:


X_train, X_test, y_train, y_test = cross_validation(train_df,
                                                    split_size=0.2,
                                                    random_state=15,
                                                    shuffle_state=True)


# In[155]:


X_train = X_train.str.split('$', expand=True)
X_test = X_test.str.split('$', expand=True)


# In[156]:


y_train.dtypes


# In[157]:


# X_test = X_test.iloc[:, :-1]
# X_test


# In[158]:


X_train.columns = cols
X_test.columns = cols


# In[159]:


y_train_df = pd.DataFrame(y_train, columns = ['label_group'])


# In[160]:


y_test_df = pd.DataFrame(y_test, columns = ['label_group'])


# In[161]:


X_train


# In[162]:


train_image_list = X_train['image'].to_list()
train_image_list


# In[170]:


y_test.count()


# In[171]:


# y_train = X_train['label_group'].to_list()
# y_train = np.asarray(y_train)
y_arr = y_train


# In[172]:


y_arr.shape


# In[173]:


# y_test = X_test['label_group'].to_list()
# y_test = np.asarray(y_test)
y_test_arr = y_test


# In[174]:


y_test_arr.shape


# In[175]:


test_image_list = X_test['image'].to_list()


# In[176]:


len(test_image_list)


# In[177]:


import cv2
res = []
base_path = 'Project/data/train_images/'
for image in train_image_list:
    image_path = base_path+image
    img = cv2.imread(image_path)
    res.append(cv2.resize(img, (224,224)))
    del img


# In[178]:


test_res = []
base_path = 'Project/data/train_images/'
for image in test_image_list:
    image_path = base_path+image
    img = cv2.imread(image_path)
    test_res.append(cv2.resize(img, (224,224)))
    del img
    


# In[179]:


x_arr = np.asarray(res)


# In[180]:


x_arr.shape


# In[181]:


x_test_arr = np.asarray(test_res)
x_test_arr.shape


# In[182]:


y_arr = y_train.to_numpy()
y_arr.shape


# In[183]:


y_test_arr = y_test.to_numpy()
y_test_arr.shape


# In[184]:


x_arr = x_arr/255


# In[185]:


x_test_arr = x_test_arr/255
x_test_arr


# # Create training label dict

# In[186]:


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


# In[187]:


y_arr.size


# In[188]:


len(modified_labels)


# In[189]:


uniqueKeys = set(my_dict.keys())
len(uniqueKeys)


# In[190]:


uniqueValues = set(my_dict.values())
len(uniqueValues)


# # Create test label dict

# In[191]:


modified_labels_test = []
for label in y_test_arr:
    if label not in my_dict.keys():
        ptr = ptr+1
        my_dict[label] = ptr
        modified_labels_test.append(ptr)
    else:
        my_dict[label] = my_dict.get(label)
        modified_labels_test.append(my_dict.get(label))


# In[192]:


len(modified_labels_test)


# In[193]:


modified_labels_test[0]


# In[194]:


modified_label_arr = np.asarray(modified_labels)
modified_label_arr


# In[195]:


modified_label_test_arr = np.asarray(modified_labels_test)
modified_label_test_arr


# # Take pre-train Tensorflow_hub efficientNet model and retrain it using train images with 375 classes

# In[196]:


feature_extractor_model = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

pretrained_model_without_top_layer = hub.KerasLayer(
    feature_extractor_model, input_shape=(224, 224, 3), trainable=False)


# In[197]:


model = tf.keras.Sequential([
  pretrained_model_without_top_layer,
  tf.keras.layers.Dense(len(uniqueKeys))
])

model.summary()


# In[198]:


X_train_scaled = x_arr


# In[199]:


y_arr


# In[200]:


model.compile(
  optimizer="adam",
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['acc'])

model.fit(X_train_scaled, modified_label_arr, epochs=5)


# In[202]:


modified_label_test_arr


# In[203]:


X_test_scaled = x_test_arr


# In[204]:


y_test_arr


# In[206]:


model.evaluate(X_test_scaled,modified_label_test_arr)


# In[ ]:




