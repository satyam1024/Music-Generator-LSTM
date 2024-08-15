import music21 as m21
from music21 import  converter, instrument, note, stream, chord, duration
import os
from midi2audio import FluidSynth
import tensorflow as tf
import numpy as np
import glob
import json
import pickle
import np_utils
keras=tf.keras
from keras.utils import to_categorical



intervals = range(1)
seq_len = 32
count=1
SAVE_DIR="output"
# model params
embed_size = 100
rnn_units = 256
use_attention = True



def load_songs(dataset_path):
    songs = []
    count=0
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:
            print(file)
            print(count)
            count=count+1
            if file.endswith("mid"):
                song = m21.converter.parse(os.path.join(path, file)).chordify()
                songs.append(song)
    return songs
  

def get_music_list(data_folder):    
    file_list = glob.glob(os.path.join(data_folder, "*.mid"))
    parser = converter    
    return file_list, parser

def get_distinct(elements):
    element_names = sorted(set(elements))
    n_elements = len(element_names)
    return (element_names, n_elements)

def create_lookups(element_names):
    element_to_int = dict((element, number) for number, element in enumerate(element_names))
    int_to_element = dict((number, element) for number, element in enumerate(element_names))
    return (element_to_int, int_to_element)      

def create_data(music_list):

    notes = []
    durations = []

    for i, song in enumerate(music_list):
        print(i+1, "Parsing %s" % song)
        print(song)
         
        for interval in intervals:
            score = song.transpose(interval)

            notes.extend(['START'] * seq_len)
            durations.extend([0]* seq_len)

            for element in score.flat:                
                if isinstance(element, note.Note):
                    if element.isRest:
                        notes.append(str(element.name))
                        durations.append(element.duration.quarterLength)
                    else:
                        notes.append(str(element.nameWithOctave))
                        durations.append(element.duration.quarterLength)
                        
                if isinstance(element, chord.Chord):
                    notes.append('.'.join(n.nameWithOctave for n in element.pitches))
                    durations.append(element.duration.quarterLength)
    
    
    return (notes,durations)

def create_seed(song):

    notes = []
    durations = []

    score = song.transpose(0)


    for element in score.flat:                
        if isinstance(element, note.Note):
            if element.isRest:
                notes.append(str(element.name))
                durations.append(element.duration.quarterLength)
            else:
                notes.append(str(element.nameWithOctave))
                durations.append(element.duration.quarterLength)
                        
        if isinstance(element, chord.Chord):
            notes.append('.'.join(n.nameWithOctave for n in element.pitches))
            durations.append(element.duration.quarterLength)
    
    return (notes,durations)

    

def prepare_sequences(notes, durations, lookups, distincts, seq_len =32):
    note_to_int, int_to_note, duration_to_int, int_to_duration = lookups
    n_notes, n_durations = distincts

    notes_network_input = []
    notes_network_output = []
    durations_network_input = []
    durations_network_output = []
  
    notes=splitter(notes)
  
    for i in range(len(notes) - seq_len):
        
        
        notes_sequence_in = []
        for char in notes[i:i + seq_len]:
            note_int_sequence = [0 for _ in range(n_notes)]
            for note_char in char:
                try:
                    note_int_sequence[(note_to_int[note_char])]=1
                except KeyError:
                   
                    print(f"Key {note_char} is missing in note_to_int dictionary.")
                    note_int_sequence.append(note_to_int['<UNK>'])
            notes_sequence_in.append(note_int_sequence)

      
        notes_sequence_out = [0 for _ in range(n_notes)]
        for note_char in notes[i + seq_len]:
            try:
                notes_sequence_out[(note_to_int[note_char])]=1
            except KeyError:
                
                print(f"Key {note_char} is missing in note_to_int dictionary.")
                notes_sequence_out.append(note_to_int['<UNK>'])  
                    
        notes_network_input.append(notes_sequence_in)
        notes_network_output.append(notes_sequence_out)
        
        durations_sequence_in = durations[i:i + seq_len]
        durations_sequence_out = durations[i + seq_len]
        durations_network_input.append([duration_to_int[char] for char in durations_sequence_in])
        durations_network_output.append(duration_to_int[durations_sequence_out])
        
    notes_network_input=np.array(notes_network_input)
    notes_network_output=np.array(notes_network_output)
    print(notes_network_input[0][0])
    print(notes_network_output[0])
    print(notes_network_input.shape)
    print(notes_network_output.shape)

    n_patterns = len(notes_network_input)
    durations_network_input = np.reshape(durations_network_input, (n_patterns, seq_len))
    network_input = [notes_network_input, durations_network_input]
    durations_network_output = to_categorical(durations_network_output, num_classes=n_durations)
    network_output = [notes_network_output, durations_network_output]
    return (network_input, network_output)


def splitter(notes):
    
    result = []
    for item in notes:
        if(item!="START"):
            splitted = item.split('.')
      
            for a in splitted:
                result.append(a) 
            result.append("_")
        else:
            result.append(item)

    return result

def convert_songs_to_int(songs,lookup):
    int_songs = []
    

    for symbol in songs:
        int_songs.append(lookup[symbol])

    return int_songs


def dconvert_songs_to_int(songs,lookup):
    int_songs = []

    # map songs to int
    for symbol in songs:
        
        int_songs.append(lookup[symbol])

    return int_songs

def generate_training_sequences(int_songs,sequence_length,classes=90):

    inputs = []
    targets = []

    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])


    vocabulary_size = len(set(int_songs))
    print(vocabulary_size)
   
    inputs = keras.utils.to_categorical(inputs, num_classes=classes)
    targets = keras.utils.to_categorical(targets, num_classes=classes)

    return inputs, targets

def preprocess(filepath):
    print("Loading songs...")
    music_list = load_songs(filepath)

    notes,durations=create_data(music_list)
    duration=[]
    print("Notes =>")
    print(notes)
    with open(os.path.join(SAVE_DIR, 'lookup'), 'rb') as f:
        lookup =pickle.load(f)
    with open(os.path.join(SAVE_DIR, 'length'), 'rb') as f:
        length =pickle.load(f)
    
    lookup[0]['_']=89
    lookup[2]['_']=len(lookup[2])
    dl=len(lookup[2])

    notes=splitter(notes)
    k=0
    i=0
    for i in range(len(notes)):
        if(notes[i]!="START"):
            if(notes[i]=='_'):
                duration.append('_')
                k+=1
            else:
                duration.append(durations[k])
        else:
            duration.append(durations[k])
            k+=1

    print(notes)
    notes=convert_songs_to_int(notes,lookup[0])
    duration=dconvert_songs_to_int(duration,lookup[2])
    with open(os.path.join(SAVE_DIR, 'seed'), 'wb') as f:
        pickle.dump([notes[16:50],durations[16:50]], f)

    note_input,note_output=generate_training_sequences(notes,32)
    duration_input,duration_output=generate_training_sequences(duration,32,dl)
    
    network_input= [note_input,duration_input]
    network_output=[note_output,duration_output]

    with open(os.path.join(SAVE_DIR, 'input'), 'wb') as f:
            pickle.dump(network_input, f)
    with open(os.path.join(SAVE_DIR, 'output'), 'wb') as f:
            pickle.dump(network_output, f)
    
    return (network_input, network_output)   
            
            


filename = 'data/bach'

def main():

    preprocess(filename)

if __name__ == "__main__":
    main()