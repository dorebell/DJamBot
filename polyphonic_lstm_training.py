# Author: Jonas Wiesendanger wjonas@student.ethz.ch
from settings import *
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation, Lambda, Concatenate, Input, Dropout
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adam
# from keras.utils import to_categorical
from keras.utils import np_utils
from keras.layers.wrappers import Bidirectional
from random import shuffle
import progressbar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np
import _pickle as pickle

import data_class
import chord_model

from keras import losses


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

# Uncomment next block if you only want to use a fraction of the GPU memory:

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#set_session(tf.Session(config=config))

def weighted_square_error(y_true, y_pred):
    
    prob_true = y_true[:, :new_num_notes]
    prob_pred = y_pred[:, :new_num_notes]
    vel_true = y_true[:, new_num_notes: 2*new_num_notes]
    vel_pred = y_pred[:, new_num_notes: 2*new_num_notes]
    ce = losses.binary_crossentropy(prob_true, prob_pred)
    vel_pair = tf.multiply(prob_true, vel_pred) + tf.multiply(1 - prob_true, vel_true) # model DeepJ's implementation
    mse = losses.mean_squared_error(vel_true, vel_pair)
    cost = ce + mse
    return cost


# Test function
def test():
    # currently this function is not really used, but its syntax is copied in the main function
    print('\nTesting:')
    total_test_loss = 0

    bar = progressbar.ProgressBar(maxval=test_set_size, redirect_stdout=False)
    for i, test_song in enumerate(test_set):
        X_test, Y_test = make_feature_vector(test_song, chord_test_set[i], chord_embed_method)
        loss = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=verbose)
        model.reset_states()
        total_test_loss += loss
        bar.update(i)
    total_test_loss_array.append(total_test_loss/test_set_size)
    print('\nTotal test loss: ', total_test_loss/test_set_size)
    print('-'*50)
    plt.plot(total_test_loss_array, 'b-')
    plt.plot(total_train_loss_array, 'r-')
    if show_plot: plt.show()
    if save_plot: plt.savefig(model_path+'plot.png')
    pickle.dump(total_test_loss_array,open(model_path+'total_test_loss_array.pickle', 'wb'))
    pickle.dump(total_train_loss_array,open(model_path+'total_train_loss_array.pickle', 'wb'))

# Make feature vectors with the notes and the chord information
def make_feature_vector(song, chords, chord_embed_method, chord_embed_model):
    if  next_chord_feature:
#        X = np.array(data_class.make_one_hot_note_vector(song[:(((len(chords)-1)*fs*2)-1)], num_notes))
        X = song[:(((len(chords)-1)*fs*2)-1)]
    else:
#        X = np.array(data_class.make_one_hot_note_vector(song[:((len(chords)*fs*2)-1)], num_notes))
        X = song[:((len(chords)*fs*2)-1)]
    print(X.shape)
    X = X[:,low_crop:high_crop]
    X[:,:,1] = X[:,:,1]/max_velocity #normalize velocity
    X = np.reshape(X, (X.shape[0], -1))
    if chord_embed_method == 'embed':
        X_chords = list(chord_embed_model.embed_chords_song(chords))
    elif chord_embed_method == 'onehot':
        X_chords = data_class.make_one_hot_vector(chords, num_chords)
    elif chord_embed_method == 'int':
        X_chords = [[x] for x in chords]
    X_chords_new = []
    Y = X[1:]
    
    for j, _ in enumerate(X):
        ind = int(((j+1)/(fs*2)))
        
        if next_chord_feature:
            ind2 = int(((j+1)/(fs*2)))+1
#            print(j)
#            print(ind, ' ', ind2)
#            print(X_chords[ind].shape)
            X_chords_new.append(list(X_chords[ind])+list(X_chords[ind2]))
        else:
            X_chords_new.append(X_chords[ind])
            
    X_chords_new = np.array(X_chords_new)
   
    X = np.append(X, X_chords_new, axis=1)
    
    if counter_feature:
        # the following works for 8th notes--modify if using other subdivisions
        counter = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
        if next_chord_feature:
            counter = np.array(counter*(len(X_chords)-1))[:-1]
        else:
            counter = np.array(counter*len(X_chords))[:-1]
        X = np.append(X, counter[:X.shape[0]], axis=1)
    X = X[:-1]
    X = np.reshape(X, (X.shape[0], 1, -1))
    
    return X, Y


def main():
    # Path to the fully trained chord model for the chord embeddings:
    chord_model_path = 'models/chords/1528360369-Shifted_True_Lr_5e-05_EmDim_10_opt_Adam_bi_False_lstmsize_512_trainsize_324_testsize_36_samples_per_bar16/model_Epoch24.pickle'
    # Path where the polyphonic models are saved:
    model_path = 'models/chords_mldy/'
    model_filetype = '.pickle'

    epochs = 20 # 100
    train_set_prop = 9
    test_set_prop = 1
    test_step = 150          # Calculate error for test set every this many songs

    verbose = False
    show_plot = False
    save_plot = True
    lstm_size = 256
    batch_size = 16
    learning_rate = 5e-06
    step_size = 1
    save_step = 1
    shuffle_train_set = True
    bidirectional = False
    embedding = False
    optimizer = 'Adam'
    
    dropout = 0.8

    print('loading data...')
    # Get Train and test sets
    train_set, test_set, chord_train_set, chord_test_set = data_class.get_note_train_and_test_set(train_set_prop, test_set_prop)
    train_set_size = len(train_set)
    test_set_size = len(test_set)
    print('Training set size: ', train_set_size)
    print('Test set size: ', test_set_size)

    fd = {'shifted': shifted, 'next_chord_feature': next_chord_feature, 'chord_embed_method': chord_embed_method, 'counter': counter_feature, 'highcrop': high_crop, 'lowcrop':low_crop, 'lr': learning_rate, 'opt': optimizer,
        'bi': bidirectional, 'lstms': lstm_size, 'trainsize': train_set_size, 'testsize': test_set_size, 'epochs': epochs}
    
    model_name = 'Shifted_%(shifted)s_NextChord_%(next_chord_feature)s_ChordEmbed_%(chord_embed_method)s_Counter_%(counter)s_Highcrop_%(highcrop)s_Lowcrop_%(lowcrop)s_Lr_%(lr)s_opt_%(opt)s_bi_%(bi)s_lstmsize_%(lstms)s_trainsize_%(trainsize)s_testsize_%(testsize)s_epochs_%(epochs)s' % fd

    model_path = model_path + model_name + '/'
    if not os.path.exists(model_path):
        os.makedirs(model_path) 
    
    if chord_embed_method == 'embed':
        chord_dim = chord_embedding_dim
    elif chord_embed_method == 'onehot':
        chord_dim = num_chords
    elif chord_embed_method == 'int':
        chord_dim = 1

    if next_chord_feature:
        chord_dim = chord_dim*2

    # Load model for chord embeddings
    chord_embed_model = chord_model.Embed_Chord_Model(chord_model_path)



    # Build Melody Model
    print('creating model...')

    inputs = Input(shape = (step_size, (new_num_notes*2)+chord_dim+counter_size))
    if bidirectional:
        lstm = Bidirectional(LSTM(lstm_size, batch_size = batch_size, return_sequences = True))(inputs)
        lstm = Dropout(dropout)(lstm)
        for l in range(num_poly_layers-1):
            lstm = Bidirectional(LSTM(lstm_size, batch_size = batch_size))(lstm)
            lstm = Dropout(dropout)(lstm)
    else:
        lstm = LSTM(lstm_size, batch_size = batch_size)(inputs)
        lstm = Dropout(dropout)(lstm)
        for l in range(num_poly_layers-1):
            lstm = LSTM(lstm_size, batch_size = batch_size)(lstm)
            lstm = Dropout(dropout)(lstm)

    note_dense = Dense(new_num_notes, activation = 'sigmoid', name = 'note_dense')(lstm)
    volume_dense = Dense(new_num_notes, name = 'volume_dense')(lstm)
    denses = Concatenate(axis = -1)([note_dense, volume_dense])
    model = Model(inputs = inputs, outputs = denses)



    #removed sigmoid activation so volumes will not be probabilities but just values - DDJZ
    if optimizer == 'RMS': optimizer = RMSprop(lr=learning_rate)
    if optimizer == 'Adam': optimizer = Adam(lr=learning_rate)
    loss = weighted_square_error  #changed from categorical_crossentropy, since output no longer probability.
    model.compile(optimizer, loss)


    # initialize loss arrays
    total_test_loss_array = [] 
    total_train_loss_array = []
    total_test_loss = 0
    total_train_loss = 0

    # Save Parameters to text file
    with open(model_path + 'params.txt', "w") as text_file:
        text_file.write("Chord Model: %s" % chord_model_path + '\n')
        text_file.write("epochs: %s" % epochs + '\n')
        text_file.write("train_set_size: %s" % train_set_size + '\n')
        text_file.write("test_set_size: %s" % test_set_size + '\n')
        text_file.write("lstm_size: %s" % lstm_size + '\n')
        text_file.write("learning_rate: %s" % learning_rate + '\n')
        text_file.write("save_step: %s" % save_step + '\n')
        text_file.write("shuffle_train_set: %s" % shuffle_train_set + '\n')
        text_file.write("test_step: %s" % test_step + '\n')
        text_file.write("bidirectional: %s" % bidirectional + '\n')
        text_file.write("num_chords: %s" % num_chords + '\n')
        text_file.write("chord_n: %s" % chord_n + '\n')

    # Train model
    print('training model...')
    for e in range(1, epochs+1):
        
        print('Epoch ', e, 'of ', epochs, 'Epochs\nTraining:')
        
        # Shuffle training set order
        if shuffle_train_set:
            print('Shuffling training set: ')
            # Zip lists together an shuffle and unzip again
            ziperoni = list(zip(train_set, chord_train_set))
            shuffle(ziperoni)
            train_set, chord_train_set = zip(*ziperoni)

        bar = progressbar.ProgressBar(maxval=train_set_size)
        
        # Train model with each song separately
        for i, song in enumerate(train_set):
            X = np.zeros(song.shape)
            Y = np.zeros(song.shape)
            X, Y = make_feature_vector(song, chord_train_set[i], chord_embed_method, chord_embed_model)
            hist = model.fit(X, Y, batch_size=batch_size, shuffle=False, verbose=verbose)
            model.reset_states()
            bar.update(i)
            total_train_loss += hist.history['loss'][0]
            if (i+1)%test_step is 0:
                total_train_loss = total_train_loss/test_step
                total_train_loss_array.append(total_train_loss)
                print('\nTotal train loss: ', total_train_loss)
                #test()
                #not the most elegant way to do this, but...

                print('\nTesting:')
                total_test_loss = 0

                bar2 = progressbar.ProgressBar(maxval=test_set_size, redirect_stdout=False)
                for j, test_song in enumerate(test_set):
                    X_test, Y_test = make_feature_vector(test_song, chord_test_set[j], chord_embed_method, chord_embed_model)
                    loss = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=verbose)
                    model.reset_states()
                    total_test_loss += loss
                    bar2.update(j)
                total_test_loss_array.append(total_test_loss/test_set_size)
                print('\nTotal test loss: ', total_test_loss/test_set_size)
                print('-'*50)
                plt.plot(total_test_loss_array, 'b-')
                plt.plot(total_train_loss_array, 'r-')
                if show_plot: plt.show()
                if save_plot: plt.savefig(model_path+'plot.png')
                pickle.dump(total_test_loss_array,open(model_path+'total_test_loss_array.pickle', 'wb'))
                pickle.dump(total_train_loss_array,open(model_path+'total_train_loss_array.pickle', 'wb'))
                total_train_loss = 0

        total_train_loss = 0
        if e%save_step is 0:
            print('\nsaving model')
            model_save_path = model_path + 'model' + 'Epoch' + str(e) + model_filetype
            model.save(model_save_path)

if __name__ == "__main__":
    main()
