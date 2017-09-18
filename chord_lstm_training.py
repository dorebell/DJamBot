# Author: Jonas Wiesendanger wjonas@student.ethz.ch
from settings import *
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
from keras.layers.wrappers import Bidirectional
from random import shuffle
import progressbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import _pickle as pickle
import os
import data_class

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# Uncomment next block if you only want to use a fraction of the GPU memory:

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#set_session(tf.Session(config=config))

#Path where the models are saved:
model_path = 'models/chords/'
model_filetype = '.h5'

epochs = 20
train_set_size = 30
test_set_size = 5
test_step = 30          # Calculate error for test set every this many songs

verbose = False
show_plot = False
save_plot = True
lstm_size = 256
batch_size = 1
learning_rate = 0.00001
step_size = 1
save_step = 1
shuffle_train_set = True
bidirectional = False
optimizer = 'Adam'

fd = {'shifted': shifted, 'lr': learning_rate, 'emdim': chord_embedding_dim, 'opt': optimizer,
'bi': bidirectional, 'lstms': lstm_size, 'trainsize': train_set_size, 'testsize': test_set_size}
model_name = 'Shifted_%(shifted)s_Lr_%(lr)s_EmDim_%(emdim)s_opt_%(opt)s_bi_%(bi)s_lstmsize_%(lstms)s_trainsize_%(trainsize)s_testsize_%(testsize)s' % fd

model_path = model_path + model_name + '/'
if not os.path.exists(model_path):
    os.makedirs(model_path) 



print('loading data...')
train_set, test_set = data_class.get_chord_train_and_test_set(train_set_size, test_set_size)
print('creating model...')
model = Sequential()
model.add(Embedding(num_chords, chord_embedding_dim, batch_size=1, input_length=1))
if bidirectional: model.add(Bidirectional(LSTM(lstm_size, stateful=True)))
else: model.add(LSTM(lstm_size, stateful=True))
model.add(Dense(num_chords))
model.add(Activation('softmax'))
if optimizer == 'Adam':
    optimizer = Adam(lr=learning_rate)
elif optimizer == 'RMS':
    optimizer = RMSprop(lr=learning_rate)
loss = 'categorical_crossentropy'
model.compile(optimizer, loss)


total_test_loss_array = [] 
total_train_loss_array = []
total_test_loss = 0

def test():
    print('\nTesting:')
    total_test_loss = 0

    bar = progressbar.ProgressBar(max_value=test_set_size, redirect_stdout=False)
    for i, test_song in enumerate(test_set):
        X_test = test_song[:-1]
        Y_test = to_categorical(test_song[1:], num_classes=num_chords)
        loss = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=verbose)
        model.reset_states()
        total_test_loss += loss
        bar.update(i)
    total_test_loss_array.append(total_test_loss/test_set_size)
    print('\nTotal test loss: ', total_test_loss/test_set_size)
    print('-'*50)
    plt.plot(total_test_loss_array, 'b-', label='test loss')
    plt.plot(total_train_loss_array, 'r-', label='train loss')
#    plt.legend()
    plt.ylabel(model_path)
#    plt.axis([0, 50, 3, 5])
    plt.grid()
    if show_plot: plt.show()
    if save_plot: plt.savefig(model_path+'plot.png')
    pickle.dump(total_test_loss_array,open(model_path+'total_test_loss_array.pickle', 'wb'))
    pickle.dump(total_train_loss_array,open(model_path+'total_train_loss_array.pickle', 'wb'))

def train():
    print('training model...')
    total_train_loss = 0
    for e in range(1, epochs+1):
        print('Epoch ', e, 'of ', epochs, 'Epochs\nTraining:')
        if shuffle_train_set:
            shuffle(train_set)
        bar = progressbar.ProgressBar(max_value=train_set_size)
        for i, song in enumerate(train_set):
            X = song[:-1]
            Y = to_categorical(song[1:], num_classes=num_chords)
            hist = model.fit(X, Y, batch_size=batch_size, shuffle=False, epochs=1, verbose=verbose)
            model.reset_states()
            bar.update(i)
#            print(hist.history)
            total_train_loss += hist.history['loss'][0]
            if (i+1)%test_step is 0:
                total_train_loss = total_train_loss/test_step
                total_train_loss_array.append(total_train_loss)
                test()
                total_train_loss = 0
    
        if e%save_step is 0:
            print('saving model')
            model_save_path = model_path + 'model_' + 'Epoch' + str(e) + model_filetype
            model.save(model_save_path)

def save_params():
    with open(model_path + 'params.txt', "w") as text_file:
        text_file.write("epochs: %s" % epochs + '\n')
        text_file.write("train_set_size: %s" % train_set_size + '\n')
        text_file.write("test_set_size: %s" % test_set_size + '\n')
        text_file.write("lstm_size: %s" % lstm_size + '\n')
        text_file.write("embedding_dim: %s" % chord_embedding_dim + '\n')
        text_file.write("learning_rate: %s" % learning_rate + '\n')
        text_file.write("save_step: %s" % save_step + '\n')
        text_file.write("shuffle_train_set: %s" % shuffle_train_set + '\n')
        text_file.write("test_step: %s" % test_step + '\n')
        text_file.write("bidirectional: %s" % bidirectional + '\n')
        text_file.write("num_chords: %s" % num_chords + '\n')
        text_file.write("chord_n: %s" % chord_n + '\n')

save_params()
train()