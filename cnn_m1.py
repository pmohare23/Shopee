#!/usr/bin/env python
import numpy as np # linear algebra
import pandas as pd # data processing
import os
import numpy as np

# In[34]:

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

import os
os.chdir("../")
print("parent path -> ", os.getcwd())


# In[35]:


train_df = pd.read_csv('Project/data/train.csv')


# In[39]:


print(train_df.columns)


# In[8]:

# combine columns of train_df to have data and label fields
train_df['data'] = train_df['posting_id'].astype(str) + "$" + train_df['image'].astype(str) + "$" + train_df['image_phash'].astype(str) + "$" + train_df['title'].astype(str)


# In[9]:


cols = ['posting_id','image','image_phash','title']
train_df = train_df.drop(cols, axis=1)


# In[10]:


print(train_df.head())


# In[11]:

# Cross validation
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


# In[20]:


X_train, X_test, y_train, y_test = cross_validation(train_df,
                                                    split_size=0.2,
                                                    random_state=15,
                                                    shuffle_state=True)


# In[21]:


X_train = X_train.str.split('$', expand=True)
X_test = X_test.str.split('$', expand=True)


# In[22]:


print(y_train.dtypes)


# In[23]:


print(X_train.head())


# In[24]:


# X_test = X_test.iloc[:, :-1]
# X_test


# In[25]:


X_train.columns = cols
X_test.columns = cols


# In[26]:


X_train.head(10)


# In[27]:


y_train_df = pd.DataFrame(y_train, columns = ['label_group'])
print(y_train_df.head(10))


# In[29]:


y_test_df = pd.DataFrame(y_test, columns = ['label_group'])
print(y_test_df.head(10))


# In[28]:


# train_data = X_train.concat(y_train_df)
train_data = pd.concat([X_train, y_train_df], axis=1)
print(train_data.head())


# In[30]:


# train_data = X_train.concat(y_train_df)
test_data = pd.concat([X_test, y_test_df], axis=1)
print(test_data.head())

train_image_list = X_train['image'].to_list()
train_image_list


# In[180]:


test_image_list = X_test['image'].to_list()


# In[181]:


print(len(train_image_list))


# In[182]:


# image resize
import cv2
res = []
base_path = 'Project/data/train_images/'
for image in train_image_list:
    image_path = base_path+image
    img = cv2.imread(image_path)
    res.append(cv2.resize(img, (224,224)))
    del img


# In[183]:


test_res = []
base_path = 'Project/data/train_images/'
for image in test_image_list:
    image_path = base_path+image
    img = cv2.imread(image_path)
    test_res.append(cv2.resize(img, (224,224)))
    del img
    


# In[211]:


x_arr = np.asarray(res)


# In[212]:


x_arr.shape


# In[213]:


x_test_arr = np.asarray(test_res)
x_test_arr.shape


# In[214]:


y_train[:5]


# In[215]:


y_arr = y_train.to_numpy()
y_arr.shape


# In[216]:


y_test[:5]
y_test_arr = y_test.to_numpy()
y_test_arr.shape


# In[217]:


# Image normalize
x_arr = x_arr/255


# In[218]:


x_test_arr = x_test_arr/255
x_test_arr


# In[219]:

# Assigning labels to train images based on label groups

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


# In[220]:


y_arr.size


# In[221]:


len(modified_labels)


# In[222]:


y_test_arr


# In[223]:


uniqueKeys = set(my_dict.keys())
len(uniqueKeys)


# In[224]:


uniqueValues = set(my_dict.values())
len(uniqueValues)


# In[225]:


my_dict


# In[226]:
# Assigning labels to test images based on label groups

modified_labels_test = []
for label in y_test_arr:
    if label not in my_dict.keys():
        ptr = ptr+1
        my_dict[label] = ptr
        modified_labels_test.append(ptr)
    else:
        my_dict[label] = my_dict.get(label)
        modified_labels_test.append(my_dict.get(label))


# In[227]:


len(modified_labels_test)


# In[228]:


modified_labels_test[0]


# In[229]:


u_value = set( val for val in my_dict.values())
# print("Unique Values: ",u_value)
len(u_value)
# u_value


# In[230]:


len(modified_labels)


# In[231]:


modified_label_arr = np.asarray(modified_labels)
modified_label_arr


# In[232]:


modified_label_test_arr = np.asarray(modified_labels_test)
modified_label_test_arr


# In[206]:


categories = np.unique(y_arr)


# In[233]:


categories.size


# # CNN model

# In[208]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(uniqueKeys), activation='softmax')
])


# In[209]:

# Hyper params

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[210]:


cnn.fit(x_arr, modified_label_arr, epochs=25)


# In[234]:


cnn.evaluate(x_test_arr,modified_label_test_arr)


# In[235]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = cnn.predict(x_test_arr)
y_pred


# In[236]:


y_pred_classes = [np.argmax(element) for element in y_pred]


# In[238]:


print("Classification Report: \n", classification_report(modified_label_test_arr, y_pred_classes))


# In[239]:

# Using f1-score for evaluating model's performance

from sklearn.metrics import f1_score
f1_score = f1_score(modified_label_test_arr, y_pred_classes, average='weighted')
print("f1_score --> ", f1_score)

# # Creating submission file

# In[70]:


y_pred_classes


# In[72]:


X_test['pred'] = y_pred_classes

submission_df = X_test
submission_df.columns


# In[84]:


drop_cols = ['image','image_phash','title']
submission_df = submission_df.drop(columns=drop_cols)

# In[85]:


submission_df['matches'] = submission_df.groupby(['pred'])['posting_id'].transform(lambda x: ' '.join(x))

submission_df.head(10)

submission_df = submission_df.drop(columns='pred')

submission_df.to_csv("Project/output/submission_cnn.csv")
