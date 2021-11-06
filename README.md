# Project: Shopee e-commerce product matching
* Entire dataset can be found [here](https://www.kaggle.com/c/shopee-product-matching/data)

## List of Models evaluated:
* Text Processing With Lemmatization
* Text Processing Without Lemmitization
* ORB+Kmeans+SVM/Random Forest
* BRISK+Kmeans+SVM/Random Forest
* ANN
* CNN
* Transfer learning/Feature vector Mobile Net 
* Transfer learning/Feature vector Efficient Net
* Transfer learning Inception V3

## Steps to run each models:
* Text Processing With Lemmatization:
#### python3 text_pred_word2Vec-cv-all-with-lemma.py

* Text Processing Without Lemmatization:
#### python3 text_pred_word2Vec-cv-all-without-lemma.py

* ORB:
#### python3 orb-1000.py

* BRISK:
#### python3 brisk-1000.py

* ANN:
#### python3 ann-m1.py

* CNN:
#### python3 cnn-m1.py

* Transfer learning MobileNet:
#### python3 transfer_learning_mobilenet.py

* Transfer learning EfficientNet:
#### python3 transfer_learning_efficientNet.py

* Inception V3:
#### Images under **data/folder** will be used to train and execute the model
#### python3 inception-subset.py

#ipynb file can be run through Jupyter notebook.