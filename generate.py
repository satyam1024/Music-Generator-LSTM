import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
keras=tf.keras
import music21 as m21
from music21 import instrument, note, stream, chord, duration
import pickle
import os

notes_temp=0.5
duration_temp = 0.8
max_extra_notes = 50
max_seq_len = 32
seq_len = 32
model_path="output/model3.h5"
SAVE_DIR="output"

class MelodyGenerator:

    def __init__(self, lookup,model_path):

        self.model_path = model_path
        self.model = load_model(model_path)
        self.note_to_int=lookup[0]
        self.int_to_note=lookup[1]
        self.duration_to_int=lookup[2]
        self.int_to_duration=lookup[3]
    
    
    
    def conv(self,note,mapp1,dur1,mapp2):

        final=[]
        final2=[]
        temp=[]
        for i in range(len(note)):
            
            if note[i]!='START':
                if note[i]=='_':
                    final.append(temp)
                    if dur1[i-1]!="_":
                        final2.append(dur1[i-1])
                    else:
                        final2.append(dur1[i-2])
                    
                    temp=[]
                else:
                    temp.append(note[i])
            
        for i in range(len(final)):
            final[i]='.'.join(final[i])
        
        result=[]
        
        for i in range(len(final)):
            if final[i]!='':
                l=[]
                l.append(final[i])
                l.append(final2[i])
                result.append(l)
        return result



    def generate_melody(self,seed,file_path, file_name):

        notes = seed[0]
        durations = seed[1]
        
 
        print(notes)
        for i in range(len(notes)):
            if(notes[i] !=89):
                notes[i]=notes[i]+1
        
        print("\n\n")
        print(notes)
                
        # print("\n\n",durations)
        prediction_output = [[],[]]
        notes_input_sequence = []
        durations_input_sequence = []
        overall_preds = []

        
        for n, d in zip(notes,durations):
            note_int = self.int_to_note[n]
            duration_int = self.duration_to_int[d]
            
            notes_input_sequence.append(n)
            durations_input_sequence.append(duration_int)
            
            prediction_output[0].append(note_int)
            prediction_output[1].append(d)
                
                
        for note_index in range(max_extra_notes):
          
            notes_input_sequence=notes_input_sequence[-32:]
            durations_input_sequence=durations_input_sequence[-32:]
          
            en = keras.utils.to_categorical(notes_input_sequence, num_classes=90)
            ed = keras.utils.to_categorical(durations_input_sequence, num_classes=48)
        
            prediction_input = [np.array([en]), np.array([ed])]
          
            
            notes_prediction, durations_prediction= self.model.predict(prediction_input, verbose=0)
            

            i1 = self._sample_with_temperature(notes_prediction[0], notes_temp)
         
            i2 = self._sample_with_temperature(durations_prediction[0], duration_temp)
        
            notes_input_sequence.append(i1)
            durations_input_sequence.append(i2)
 

            note_result = self.int_to_note[i1]
            duration_result = self.int_to_duration[i2]

            prediction_output[0].append(note_result)
            prediction_output[1].append(duration_result)

        
        print(prediction_output[0])
        print(prediction_output[1])
        print(len(prediction_output))
        print('OK')
   
        print('Generated sequence of {} notes'.format(len(prediction_output[0])))
        melody= self._create_melody(prediction_output,file_path, file_name)
        return melody


    def _create_melody(self,output,file_path, file_name):
 
        song= self.conv(output[0],self.int_to_note,output[1],self.int_to_duration)
        
        midi_stream = stream.Stream()
        print("NOW=>")
        for pattern in song:
            print(pattern)
            note_pattern, duration_pattern = pattern
            if ('.' in note_pattern):
                notes_in_chord = note_pattern.split('.')
                chord_notes = []
                for current_note in notes_in_chord:
                    new_note = note.Note(current_note)
                    new_note.duration = duration.Duration(duration_pattern)
                    new_note.storedInstrument = instrument.Violoncello()
                    chord_notes.append(new_note)
                new_chord = chord.Chord(chord_notes)
                midi_stream.append(new_chord)
            elif note_pattern == 'rest':
                new_note = note.Rest()
                new_note.duration = duration.Duration(duration_pattern)
                new_note.storedInstrument = instrument.Violoncello()
                midi_stream.append(new_note)
            elif note_pattern != 'START':
                new_note = note.Note(note_pattern)
                new_note.duration = duration.Duration(duration_pattern)
                new_note.storedInstrument = instrument.Violoncello()
                midi_stream.append(new_note)

        midi_stream = midi_stream.chordify()
        midi_stream.write('midi', fp=os.path.join(file_path, file_name))
        
        
        
        
    def _sample_with_temperature(self, probabilites, temperature):

        predictions = np.log(probabilites) / temperature
        
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))
       
        choices = range(len(probabilites))
        index = np.random.choice(choices, p=probabilites)
        return index

    
     
if __name__ == "__main__":
    
    with open(os.path.join(SAVE_DIR, 'lookup'), 'rb') as f:
        lookup= pickle.load(f)
    with open(os.path.join(SAVE_DIR, 'seed'), 'rb') as f:
        seed= pickle.load(f)  
    lookup[0]['_']=89
    lookup[2]['_']=len(lookup[2])
    lookup[1][89]="_"
    lookup[3][47]="_"
    
   

    md=MelodyGenerator(lookup,model_path)
    melody= md.generate_melody(seed,SAVE_DIR,"new_music2.mid")
    