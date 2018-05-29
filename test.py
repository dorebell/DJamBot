
from keras.models import Sequential
from keras.layers.recurrent import LSTM

model = Sequential()

model.add(LSTM(512, input_shape = [1,2,3]))