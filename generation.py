from settings import *
from keras.models import load_model
import numpy as np
from numpy import array
import _pickle as pickle
import os

import data_processing
import chord_model
import midi_functions as mf
import data_class

import tensorflow as tf

#import keras.backend as K

from polyphonic_lstm_training import weighted_square_error

chord_model_folder = 'models/chords/1528360369-Shifted_True_Lr_5e-05_EmDim_10_opt_Adam_bi_False_lstmsize_512_trainsize_324_testsize_36_samples_per_bar16/'
chord_model_name = 'model_Epoch24.pickle'

melody_model_folder = 'models/chords_mldy/lstm_2_dropout_0.8_learning_rate_5e-06_lstm_size_256_bidir/'
melody_model_name = 'modelEpoch10.pickle'

midi_save_folder = 'predicted_midi/'

# changed seed_path to be pianoroll folder
#seed_path = 'data/' + shift_folder + 'indroll/'
seed_path = note_folder
seed_chord_path = 'data/' + shift_folder + 'chord_index/'

seed_name = 'Jia02.mid.pickle'


# Parameters for song generation:
BPM = 100
note_cap = 5
chord_temperature = 1

# Params for seed:
# length of the predicted song in bars:
num_bars =64
# The first seed_length number of bars from the seed will be used: 
seed_length = 4

#pred_song_length = 8*16-seed_length


with_seed = True

chord_to_index, index_to_chord = data_processing.get_chord_dict()

    
def sample_probability_vector(prob_vector):
    # Sample a probability vector, e.g. [0.1, 0.001, 0.5, 0.9]
            
    sum_probas = sum(prob_vector)
    
    
    if sum_probas > note_cap:
        prob_vector = (prob_vector/sum_probas)*note_cap
    
    note_vector = np.zeros((prob_vector.size), dtype=np.int8)
    for i, prob in enumerate(prob_vector):
        note_vector[i] = np.random.multinomial(1, [1 - prob, prob])[1]
    return note_vector

def notes_from_model(output):
    probs = output[0,:new_num_notes]
    #print("sum of probs = ",np.sum(probs))
    vels = output[0, new_num_notes: 2*new_num_notes]
    #print(vels)
    #print("average of vels = ",np.average(vels))
    vels = np.maximum(vels, 0) # because velocities are floored at 0
    vels = np.minimum(vels, 1) # because normalized velocities are capped at 1
    notes = np.multiply(sample_probability_vector(probs), vels)
    return probs, vels, notes

sd = pickle.load(open(seed_path+seed_name, 'rb'))

seed_chords = pickle.load(open(seed_chord_path+seed_name, 'rb'))[:seed_length]

seed = sd[:2*fs*seed_length, low_crop:high_crop]

seed[:,:,1] = seed[:,:,1]/max_velocity #normalize velocity
seed = np.reshape(seed, (seed.shape[0],-1))

print('loading polyphonic model ...')
melody_model = load_model(melody_model_folder+melody_model_name, custom_objects = {'weighted_square_error' : weighted_square_error})
melody_model.reset_states()

ch_model = chord_model.Chord_Model(
        chord_model_folder+chord_model_name,
        prediction_mode='sampling',
        first_chords=seed_chords,
        temperature=chord_temperature)

chords = []

for i in range((num_bars+2)):
    ch_model.predict_next()


if chord_embed_method == 'embed':
    embedded_chords = ch_model.embed_chords_song(ch_model.song)
elif chord_embed_method == 'onehot':
    embedded_chords = data_class.make_one_hot_vector(ch_model.song, num_chords)
elif chord_embed_method == 'int':
    embedded_chords = [[x] for x in ch_model.song]


chords = []

for j in range((len(ch_model.song)-2)*fs*2):
    ind = int(((j+1)/(fs*2)))
    if next_chord_feature:
        ind2 = int(((j+1)/(fs*2)))+1
        chords.append(list(embedded_chords[ind])+list(embedded_chords[ind2]))
    else:
        chords.append(embedded_chords[ind])

chords=np.array(chords)

if counter_feature:
    counter = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
    counter = np.array(counter*2*(len(ch_model.song)-2)) #ad hoc
    chords = np.append(chords, counter[:chords.shape[0]], axis=1) #ad hoc


seed = np.append(seed, chords[:seed.shape[0]], axis=1)

seed = np.reshape(seed, (seed.shape[0], 1, 1, seed.shape[1]))

next_step = None

for step in seed:
    
    next_step = melody_model.predict(step)


probs, vels, notes = notes_from_model(next_step)
#notes = np.reshape(notes, (notes.shape[0], -1, 2))


rest = []
rest.append(notes)


for chord in chords[seed.shape[0]:]:
    next_input = np.concatenate((probs, vels, chord), axis=0)
    next_input = np.reshape(next_input, (1, 1, next_input.shape[0]))
    next_step = melody_model.predict(next_input)
    probs, vels, notes = notes_from_model(next_step)
    rest.append(notes)
    
rest = np.multiply(rest, max_velocity) # normalized velocities
rest = rest.astype(int) # velocities are ints

rest = np.array(rest)
rest = np.pad(rest, ((0,0),(low_crop,num_notes-high_crop)), mode='constant', constant_values=0)
ind = np.nonzero(rest)


if not os.path.exists(midi_save_folder):
    os.makedirs(midi_save_folder)
pickle.dump(rest.T,open(midi_save_folder+'output_pianoroll.pickle', 'wb'))

instrument_names = ['Electric Guitar (jazz)', 'Acoustic Grand Piano',
'Bright Acoustic Piano', 'Electric Piano 1', 'Electric Piano 2', 'Drawbar Organ',
'Rock Organ', 'Church Organ', 'Reed Organ', 'Cello', 'Viola', 'Honky-tonk Piano', 'Glockenspiel',
'Percussive Organ', 'Accordion', 'Acoustic Guitar (nylon)', 'Acoustic Guitar (steel)', 'Electric Guitar (clean)',
'Electric Guitar (muted)', 'Overdriven Guitar', 'Distortion Guitar', 'Tremolo Strings', 'Pizzicato Strings',
'Orchestral Harp', 'String Ensemble 1', 'String Ensemble 2', 'SynthStrings 1', 'SynthStrings 2']

for instrument_name in instrument_names:
    
    mf.pianoroll_to_midi_continous(rest, midi_save_folder, instrument_name, instrument_name, BPM)
#    mf.pianoroll_to_midi(rest, 'test/midi/', instrument_name, instrument_name, BPM)

# obsolete function from JamBot
'''def ind_to_onehot(ind):
    onehot = np.zeros((len(ind), num_notes))
    for i, step in enumerate(ind):
        for note in step:
            onehot[i,note]=1
    return onehot
'''