#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import numpy as np

# In[4]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# In[6]:


import os
os.chdir("../")
print("parent path -> ", os.getcwd())


# In[7]:


train_df = pd.read_csv('Project/data/train.csv')


# In[8]:


train_df.columns


# In[9]:


train_df['data'] = train_df['posting_id'].astype(str) + "$" + train_df['image'].astype(str) + "$" + train_df['image_phash'].astype(str) + "$" + train_df['title'].astype(str)


# In[10]:


cols = ['posting_id','image','image_phash','title']
train_df = train_df.drop(cols, axis=1)


# In[11]:


train_df


# In[12]:


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


# In[13]:


X_train, X_test, y_train, y_test = cross_validation(train_df,
                                                    split_size=0.2,
                                                    random_state=15,
                                                    shuffle_state=True)


# In[14]:


X_train = X_train.str.split('$', expand=True)
X_test = X_test.str.split('$', expand=True)


# In[15]:


y_train.dtypes


# In[16]:


X_train


# In[21]:


X_test = X_test.iloc[:, :-1]
X_test


# In[22]:


X_train.columns = cols
X_test.columns = cols


# In[23]:


X_train.head(10)


# In[24]:


y_train_df = pd.DataFrame(y_train, columns = ['label_group'])
y_train_df


# In[25]:


y_test_df = pd.DataFrame(y_test, columns = ['label_group'])
y_test_df


# In[26]:


# train_data = X_train.concat(y_train_df)
train_data = pd.concat([X_train, y_train_df], axis=1)
train_data


# In[27]:


# train_data = X_train.concat(y_train_df)
test_data = pd.concat([X_test, y_test_df], axis=1)
test_data


# In[31]:


train_data.to_csv("Project/data/folder/train_cv_data.csv")


# In[32]:


test_data.to_csv("Project/data/folder/test_cv_data.csv")

train_image_list = X_train['image'].to_list()


# In[30]:


test_image_list = X_test['image'].to_list()


# In[31]:


len(train_image_list)


# In[32]:


import cv2
res = []
base_path = 'Project/data/train_images/'
for image in train_image_list:
    image_path = base_path+image
    img = cv2.imread(image_path)
    res.append(cv2.resize(img, (224,224)))
    del img


# In[33]:


test_res = []
base_path = 'Project/data/train_images/'
for image in test_image_list:
    image_path = base_path+image
    img = cv2.imread(image_path)
    test_res.append(cv2.resize(img, (224,224)))
    del img

# In[34]:


x_arr = np.asarray(res)


# In[35]:


x_arr.shape


# In[36]:


x_test_arr = np.asarray(test_res)
x_test_arr.shape


# In[37]:


y_train[:5]


# In[38]:


y_arr = y_train.to_numpy()
y_arr.shape


# In[39]:


y_test[:5]
y_test_arr = y_test.to_numpy()
y_test_arr.shape


# In[40]:


x_arr = x_arr/255


# In[41]:


x_test_arr = x_test_arr/255


# In[42]:


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


# In[43]:


y_arr.size


# In[44]:


len(modified_labels)


# In[45]:


y_test_arr


# In[46]:


uniqueKeys = set(my_dict.keys())
len(uniqueKeys)


# In[47]:


uniqueValues = set(my_dict.values())
len(uniqueValues)


# In[48]:


modified_labels_test = []
for label in y_test_arr:
    if label not in my_dict.keys():
        ptr = ptr+1
        my_dict[label] = ptr
        modified_labels_test.append(ptr)
    else:
        my_dict[label] = my_dict.get(label)
        modified_labels_test.append(my_dict.get(label))


# In[49]:


len(modified_labels_test)


# In[50]:


modified_labels_test[0]


# In[51]:


u_value = set( val for val in my_dict.values())
# print("Unique Values: ",u_value)
len(u_value)
# u_value


# In[52]:


len(modified_labels)


# In[53]:


modified_label_arr = np.asarray(modified_labels)
modified_label_arr


# In[54]:


modified_label_test_arr = np.asarray(modified_labels_test)
modified_label_test_arr


# In[55]:


categories = np.unique(y_arr)


# In[56]:


categories.size


# # ANN model

# In[58]:


ann = models.Sequential([
        layers.Flatten(input_shape=(224,224,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(11014, activation='sigmoid')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(x_arr, modified_label_arr, epochs=5)


# In[59]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(x_test_arr)
y_pred_classes = [np.argmax(element) for element in y_pred]


# In[60]:


np.unique(y_pred_classes)


# In[61]:


modified_label_test_arr


# In[62]:


print("Classification Report: \n", classification_report(modified_label_test_arr, y_pred_classes))


# In[ ]:


ann.evaluate(x_test_arr,modified_label_test_arr)


# In[ ]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(x_test_arr)


# In[ ]:


y_pred_classes = [np.argmax(element) for element in y_pred]


# In[238]:


print("Classification Report: \n", classification_report(modified_label_test_arr, y_pred_classes))


# In[239]:


from sklearn.metrics import f1_score
f1_score = f1_score(modified_label_test_arr, y_pred_classes, average='weighted')
print("f1_score --> ", f1_score)


# Creating output submission file

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

submission_df.to_csv("Project/output/submission_ann.csv")




