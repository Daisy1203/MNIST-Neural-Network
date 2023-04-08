#!/usr/bin/env python
# coding: utf-8

# In[3]:




# In[4]:


import numpy as np
from mnist import MNIST
from sklearn.model_selection import train_test_split


# In[5]:


def load_mnist_data():
    mndata = MNIST('/Users/daisy/Desktop/MNIST')
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    train_images = np.array(train_images) / 255.0
    test_images = np.array(test_images) / 255.0

    train_images = train_images.reshape(-1, 28 * 28)
    test_images = test_images.reshape(-1, 28 * 28)

    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)
    train_labels = np.array(train_labels, dtype=np.int64)
    val_labels = np.array(val_labels, dtype=np.int64)

    return train_images, train_labels, val_images, val_labels, test_images, test_labels


# In[ ]:




