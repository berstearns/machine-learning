from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Embedding
from keras.preprocessing import sequence
from keras.models import load_model

import h5py
import os
import re

model = load_model("mymodel.h5")
