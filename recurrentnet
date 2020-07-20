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
###
import numpy as np
np_load_old = np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True)

