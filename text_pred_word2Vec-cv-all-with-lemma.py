#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing

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

import os
os.chdir("../")
print("parent path -> ", os.getcwd())

# In[3]:

train_df = pd.read_csv('Project/data/train.csv')

# # Data prep for cross validation

# In[4]:


train_df['data'] = train_df['posting_id'].astype(str) + "$" + train_df['image'].astype(str) + "$"+ train_df['image_phash'].astype(str) + "$" + train_df['title'].astype(str)
cols = ['posting_id','image','image_phash','title']
train_df = train_df.drop(cols, axis=1)


# # Cross validation

# In[5]:


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


# In[6]:


X_train, X_test, y_train, y_test = cross_validation(train_df,
                                                    split_size=0.2,
                                                    random_state=15,
                                                    shuffle_state=True)


# In[7]:


X_train = X_train.str.split('$', expand=True)
X_test = X_test.str.split('$', expand=True)


# In[8]:


X_test = X_test.iloc[:, :-1]


# In[9]:


X_train.columns = cols
X_test.columns = cols


# In[10]:


train_title_list = X_train['title'].tolist()


# In[11]:


train_label_list = np.asarray(y_train)
train_label_list = list(train_label_list)


# In[12]:


test_title_list = X_test['title'].tolist()


# In[15]:


from nltk.corpus import stopwords
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for word in gensim.utils.simple_preprocess(text):
        if word not in set(stopwords.words('english')):
            result.append(lemmatize_stemming(word))
    return result


# # Process title field for train df

# In[16]:


trained_docs = X_train['title'].map(preprocess)
trained_docs =list(trained_docs)


# In[17]:


trained_docs


# In[24]:


def word2vec_model():
    word2vec_model = Word2Vec(min_count=1,
                     window=3,
                     size=50,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20)
    
    word2vec_model.build_vocab(trained_docs)
    word2vec_model.train(trained_docs,\
                         total_examples=word2vec_model.corpus_count,\
                         epochs=300,\
                         report_delay=1)
    
    return word2vec_model


# In[25]:


word2vec_model = word2vec_model()


# # Getting embedding vector

# In[26]:


embedding_vector = word2vec_model.wv
model = embedding_vector


# # Finding similarity between two vector using cosine similarity

# In[27]:


sentence_vector1_arr = []
for train_data in train_title_list:
    pre_processed_sentence1 = preprocess(train_data)
    sentence_vector1 = np.zeros(50)
    for word in pre_processed_sentence1:
        sentence_vector1 = np.add(sentence_vector1, model[word])
    sentence_vector1_arr.append(sentence_vector1)


# In[28]:


len(sentence_vector1_arr)


# In[30]:


from datetime import datetime
start_time = datetime.now()
pred_df = []
pcounter = 0
model = embedding_vector
    
for test_data in test_title_list:
    sim_score = {}
    pre_processed_sentence = preprocess(test_data)
    sentence_vector = np.zeros(50)
    for word in pre_processed_sentence:
        if word in model.vocab:
            sentence_vector = np.add(sentence_vector, model[word])
    index = 0
    for train_data in train_title_list:
        sentence_vector1 = sentence_vector1_arr[index]
        distance = dot(sentence_vector1,sentence_vector)/(norm(sentence_vector1)*norm(sentence_vector))
#         if distance > 0.6:
        sim_score[train_data] = distance
        index = index+1
    
    sorted_dict = {k: v for k, v in sorted(sim_score.items(), key=lambda item: item[1], reverse=True)}

    first = list(sorted_dict.keys())[0]
#     print("pcounter ", pcounter)
    print("sim_score key ", first)   
    print("test_data ", test_data)
    index = train_title_list.index(first)
    pred_df.append(train_label_list[index])
    pcounter = pcounter+1
end_time = datetime.now()
print('Program took : {}'.format(end_time - start_time))


# In[31]:


y_pred_classes = np.asarray(pred_df)
y_pred_classes


# In[32]:


len(y_pred_classes)


# In[33]:


label_test_arr = list(y_test)
label_test_arr = np.asarray(y_test)
label_test_arr


# In[34]:


from sklearn.metrics import confusion_matrix , classification_report, accuracy_score
print("Classification Report: \n", classification_report(label_test_arr, y_pred_classes))


# In[35]:


print("Accuracy:",accuracy_score(label_test_arr, y_pred_classes))


# # Creating submission file


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

submission_df.to_csv("Project/output/submission_text_lemma.csv")
