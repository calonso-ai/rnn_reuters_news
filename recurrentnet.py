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

# Through the following code a rnn is created with LSTM cells.
#First it´s created an embedding layer. The main objective of this layer is to assign words with
# similar meanings close values, and to words with different meanings, different values. The first
#argument, input_dim, takes the value num_word, which corresponds to the number of words that we are going to use inside
#our vocabulary or dictionary. The output_dim attribute indicates the dimension of the embedding layer (10
#in this case) and, finally, through parameter input_lenght the length of the input sequence is indicated,
#that as indicated in the statement is 30 words.
model = Sequential ()
model.add (Embedding (input_dim = num_words, output_dim = 10, input_length = 30))
#Next a layer of LSTM with 10 units is added
model.add (LSTM (10))
#Finally, a Dense layer with 46 units is added as output, since the classification will have to be done
#among 46 categories.
model.add (Dense (46, activation = 'softmax'))

#Since the output vector is defined in 46 categories, we encode the output vector to convert it
#in a one-hot vector.
one_hot_train_labels = tf.keras.utils.to_categorical (reuters_train_y)
one_hot_test_labels = tf.keras.utils.to_categorical (reuters_test_y)

#Rename the train set vectors
x_train = reuters_train_x_30
y_train = one_hot_train_labels

#Rename test set vectors
x_test = reuters_test_x_30
y_test = one_hot_test_labels

#Recurrimos al optimizador RMSProp, que implementa un descenso de gradiente como método de optimización y nos 
#permite usar un learning rate adaptativo.
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
#The model is trained during 20 epochs using test set as validation_data
hist_model = model.fit(x_train, y_train, validation_data=(x_test,y_test),epochs=20)

#Step 3: Deep LSTM network implementation

#In this section different neural networks with two LSTM layers are implemented. All networks will have exactly 
#the same architecture and number of parameters, but they will be them with sequences of different lengths (10, 20, 30, 50, 100, 150, 200 y 250.). 
#This will allow analysis of the length effect on the news rankings accuracy.

#The architecture used is as follows:
  #-Input layer with sequences of length L.
  #-Embedding layer with 32 units.
  #-LSTM layer with 32 units.
  #-LSTM layer with 32 units.
  #-Fully connected layer with 46 neurons.
  
#First, let's define a function that creates a deep LSTM network
def create_rnn (L):
     model = Sequential ()
     #An embedding layer is added, in this case with output_dim = 32 units and the input_lenght attribute will depend on the
     #value of L, which will be the length of the text for each neural network.
     model.add (Embedding (input_dim = num_words, output_dim = 32, input_length = L))
     #Then add two layers of LSTM with 32 units each
     model.add (LSTM (32, return_sequences = True))
     model.add (LSTM (32))
     #Finally, a Dense layer with 46 units is added as output, since the classification will have to be done
     #among 46 categories.
     model.add (Dense (46, activation = 'softmax'))
     #Return the created model
     return model
    
#Deep LSTM neural network with text length = 10
L1 = 10
reuters_train_x_10 = tf.keras.preprocessing.sequence.pad_sequences (reuters_train_x, maxlen = L1)
reuters_test_x_10 = tf.keras.preprocessing.sequence.pad_sequences (reuters_test_x, maxlen = L1)

model_rnn1 = create_rnn (L1)
model_rnn1.compile (optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['acc'])
hist_model1 = model_rnn1.fit (reuters_train_x_10, y_train, validation_data = (reuters_test_x_10, y_test), epochs = 20)

#Deep LSTM neural network with text length = 20
L2=20
reuters_train_x_20 = tf.keras.preprocessing.sequence.pad_sequences(reuters_train_x, maxlen=L2)
reuters_test_x_20 = tf.keras.preprocessing.sequence.pad_sequences(reuters_test_x, maxlen=L2)

model_rnn2=create_rnn(L2)
model_rnn2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
hist_model2 = model_rnn2.fit(reuters_train_x_20, y_train, validation_data=(reuters_test_x_20,y_test),epochs=20)

#Deep LSTM neural network with text length = 30
L3=30
reuters_train_x_30_bis = tf.keras.preprocessing.sequence.pad_sequences(reuters_train_x, maxlen=L3)
reuters_test_x_30_bis = tf.keras.preprocessing.sequence.pad_sequences(reuters_test_x, maxlen=L3)

model_rnn3=create_rnn(L3)
model_rnn3.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
hist_model3 = model_rnn3.fit(reuters_train_x_30_bis, y_train, validation_data=(reuters_test_x_30_bis,y_test),epochs=20)

#Deep LSTM neural network with text length = 50
L4=50
reuters_train_x_50 = tf.keras.preprocessing.sequence.pad_sequences(reuters_train_x, maxlen=L4)
reuters_test_x_50 = tf.keras.preprocessing.sequence.pad_sequences(reuters_test_x, maxlen=L4)

model_rnn4=create_rnn(L4)
model_rnn4.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
hist_model4 = model_rnn4.fit(reuters_train_x_50, y_train, validation_data=(reuters_test_x_50,y_test),epochs=20)

#Deep LSTM neural network with text length = 100
L5=100
reuters_train_x_100 = tf.keras.preprocessing.sequence.pad_sequences(reuters_train_x, maxlen=L5)
reuters_test_x_100 = tf.keras.preprocessing.sequence.pad_sequences(reuters_test_x, maxlen=L5)

model_rnn5=create_rnn(L5)
model_rnn5.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
hist_model5 = model_rnn5.fit(reuters_train_x_100, y_train, validation_data=(reuters_test_x_100,y_test),epochs=20)

#Deep LSTM neural network with text length = 150
L6=150
reuters_train_x_150 = tf.keras.preprocessing.sequence.pad_sequences(reuters_train_x, maxlen=L6)
reuters_test_x_150 = tf.keras.preprocessing.sequence.pad_sequences(reuters_test_x, maxlen=L6)

model_rnn6=create_rnn(L6)
model_rnn6.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
hist_model6 = model_rnn6.fit(reuters_train_x_150, y_train, validation_data=(reuters_test_x_150,y_test),epochs=20)

#Deep LSTM neural network with text length = 200
L7=200
reuters_train_x_200 = tf.keras.preprocessing.sequence.pad_sequences(reuters_train_x, maxlen=L7)
reuters_test_x_200 = tf.keras.preprocessing.sequence.pad_sequences(reuters_test_x, maxlen=L7)

model_rnn7=create_rnn(L7)
model_rnn7.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
hist_model7 = model_rnn7.fit(reuters_train_x_200, y_train, validation_data=(reuters_test_x_200,y_test),epochs=20)

#Deep LSTM neural network with text length = 250
L8=250
reuters_train_x_250 = tf.keras.preprocessing.sequence.pad_sequences(reuters_train_x, maxlen=L8)
reuters_test_x_250 = tf.keras.preprocessing.sequence.pad_sequences(reuters_test_x, maxlen=L8)

model_rnn8=create_rnn(L8)
model_rnn8.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
hist_model8 = model_rnn8.fit(reuters_train_x_250, y_train, validation_data=(reuters_test_x_250,y_test),epochs=20)

#Accuracy evolution for different models
plt.title('Accuracy using Test Set')
plt.plot(hist_model1.history['val_acc'], color='blue', label='Lenght sequence=10')
plt.plot(hist_model2.history['val_acc'], color='orange', label='Lenght sequence=20')
plt.plot(hist_model3.history['val_acc'], color='red', label='Lenght sequence=30')
plt.plot(hist_model4.history['val_acc'], color='green', label='Lenght sequence=50')
plt.plot(hist_model5.history['val_acc'], color='yellow', label='Lenght sequence=100')
plt.plot(hist_model6.history['val_acc'], color='black', label='Lenght sequence=150')
plt.plot(hist_model7.history['val_acc'], color='brown', label='Lenght sequence=200')
plt.plot(hist_model8.history['val_acc'], color='violet', label='Lenght sequence=250')
plt.legend(['Lenght sequence=10', 'Lenght sequence=20','Lenght sequence=30','Lenght sequence=50','Lenght sequence=100'
           ,'Lenght sequence=150','Lenght sequence=200','Lenght sequence=250'])
plt.rcParams["figure.figsize"] = [15,10]
plt.show()

#Loss function evolution for different models
plt.title('Loss using Test Set')
plt.plot(hist_model1.history['val_loss'], color='blue', label='Lenght sequence=10')
plt.plot(hist_model2.history['val_loss'], color='orange', label='Lenght sequence=20')
plt.plot(hist_model3.history['val_loss'], color='red', label='Lenght sequence=30')
plt.plot(hist_model4.history['val_loss'], color='green', label='Lenght sequence=50')
plt.plot(hist_model5.history['val_loss'], color='yellow', label='Lenght sequence=100')
plt.plot(hist_model6.history['val_loss'], color='black', label='Lenght sequence=150')
plt.plot(hist_model7.history['val_loss'], color='brown', label='Lenght sequence=200')
plt.plot(hist_model8.history['val_loss'], color='violet', label='Lenght sequence=250')
plt.legend(['Lenght sequence=10', 'Lenght sequence=20','Lenght sequence=30','Lenght sequence=50','Lenght sequence=100'
           ,'Lenght sequence=150','Lenght sequence=200','Lenght sequence=250'])
plt.rcParams["figure.figsize"] = [15,10]

plt.show()

#Step 4: Model optimization

#The goal is to find the best architecture and hyperparameters to optimize the model accuracy (our goal is to reach 70% of accuracy). 
#The points that we want to take into account when searching for the best model are the following:
  #-Recurring cell type: LSTM or GRU.
  #-Number of recurring layers.
  #-Number of units in each recurring layer.
  #-The number of embedding layer dimensions.
  #-Length of input streams.
  
def create_rnn_opt (emb_dim, L, cell_flag, cell_dim, layer_num):
    # Through the create_rnn_opt function we create a recurrent neural network in function of input parameters which
    # are passed as argument. Following are described the parameter:
    # emb_dim = Number of dimensions of the embedding layer.
    # L = Length of input streams.
    # cell_flag = Recurrent cell type: LSTM or GRU. This parameter takes the value 1 if we want to work with cells
    #LSTM or the value 0 if we want to work with GRU cells.
    # cell_dim = Number of units in each recurrent layer.
    # layer_num = Number of recurrent layers.
    model = Sequential ()
    model.add (Embedding (input_dim = num_words, output_dim = emb_dim, input_length = L))
    # Condition to create a model with cells LSTM (cell_flag = 1) or GRU (cell_flag = 0)
    if cell_flag == 1:
        if layer_num == 1:
            model.add (LSTM (cell_dim))
        elif layer_num == 2:
            model.add (LSTM (cell_dim, return_sequences = True))
            model.add (LSTM (cell_dim))
        elif layer_num == 3:
            model.add (LSTM (cell_dim, return_sequences = True))
            model.add (LSTM (cell_dim, return_sequences = True))
            model.add (LSTM (cell_dim))
    else:
        if layer_num == 1:
            model.add (GRU (cell_dim))
        elif layer_num == 2:
            model.add (GRU (cell_dim, return_sequences = True))
            model.add (GRU (cell_dim))
        elif layer_num == 3:
            model.add (GRU (cell_dim, return_sequences = True))
            model.add (GRU (cell_dim, return_sequences = True))
            model.add (GRU (cell_dim))
        
    #Finally, a Dense layer with 46 units is added as output, since the classification will have to be done
    #among 46 categories.
    model.add (Dense (46, activation = 'softmax'))
    
    return model
  
#Optimization 1: 40 embedding layer dimensions, sequence of 250 words as input, 40 LSTM cells and 3 recurrent layers.
model_rnn_opt1=create_rnn_opt(40,250,1,40,3)
model_rnn_opt1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
hist_model_opt1 = model_rnn_opt1.fit(reuters_train_x_250, y_train, validation_data=(reuters_test_x_250,y_test),epochs=20)

#Optimization 2: designed with GRU cells, 40 embedding layer dimensions, sequence of 250 words as input, 40 GRU cells and 2 recurrent layers.
model_rnn_opt2=create_rnn_opt(40,250,0,40,2)
model_rnn_opt2.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
hist_model_opt2 = model_rnn_opt2.fit(reuters_train_x_250, y_train, validation_data=(reuters_test_x_250,y_test),epochs=20)

#Optimization 3: since the results with GRU cells are good, a third optimized model is designed, but this
#time with a sequence length of 300 words as input, instead of 250.
L9=300
reuters_train_x_300 = tf.keras.preprocessing.sequence.pad_sequences(reuters_train_x, maxlen=L9)
reuters_test_x_300 = tf.keras.preprocessing.sequence.pad_sequences(reuters_test_x, maxlen=L9)
model_rnn_opt3=create_rnn_opt(40,L9,0,40,2)
model_rnn_opt3.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
hist_model_opt3 = model_rnn_opt3.fit(reuters_train_x_300, y_train, validation_data=(reuters_test_x_300,y_test),epochs=20)


