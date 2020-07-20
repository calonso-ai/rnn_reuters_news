#Libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Embedding


%matplotlib inline

###
# Workaround per ValueError: Object arrays cannot be loaded when allow_pickle=False
# Only execute once per session.
import numpy as np
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True)
###

#Step 1: Data inizialization and loading

num_words = 2000
(reuters_train_x, reuters_train_y), (reuters_test_x, reuters_test_y) = tf.keras.datasets.reuters.load_data(num_words=num_words)

n_labels = np.unique(reuters_train_y).shape[0]
print("Número de etiquetas: {}".format(n_labels))

# Printing first dataset new
print(reuters_train_x[0])

#Step 2: Simple LSTM implementation. 

#In this section it´s implemented a recurring neural network of LSTM type with a single layer. Network input is the words encoded in integer values, 
#and it´s added an embedding layer so the network learns a dense representation of the words, useful in the classification task.
#To work with a recurrent neural network it is common to define a fixed length for all texts. Texts longer than the preset length will be trimmed, 
#and shorter texts will be widden with a default value, which is often called 'pad'. The following code uses Keras preprocessing functions to 
#prepare text sequences with a sequence length of 30 words.

reuters_train_x_30 = tf.keras.preprocessing.sequence.pad_sequences(reuters_train_x, maxlen=30)
reuters_test_x_30 = tf.keras.preprocessing.sequence.pad_sequences(reuters_test_x, maxlen=30)
