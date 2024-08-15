import tensorflow as tf
keras=tf.keras
import os
import pickle
from tensorflow.keras import backend as K

SAVE_DIR="output"

# model params

embed_size = 100
rnn_units = 256
SEQUENCE_LENGTH=32
NUM_UNITS = [256]
LOSS = 'categorical_crossentropy'
LEARNING_RATE = 0.0007
EPOCHS = 1
BATCH_SIZE = 64
SAVE_MODEL_PATH = "output/model3.h5"



def build_model(notes_units, duration_units, num_units, learning_rate, embed_size=embed_size):

    note_input1 = keras.layers.Input(shape=(None, notes_units), name='note_input')
    duration_input1 = keras.layers.Input(shape=(None, duration_units), name='duration_input')

    x2 = keras.layers.Concatenate(name='concat_layer')([note_input1, duration_input1])
    lstm_layer1 = keras.layers.Bidirectional(keras.layers.LSTM(num_units,return_sequences=True))(x2)
    
    x = keras.layers.LSTM(rnn_units, return_sequences=False)(lstm_layer1)

    
    
    x1 = keras.layers.Dropout(0.2, name='dropout_layer')(x)
    
    notes_out = keras.layers.Dense(notes_units, activation='softmax', name='pitch')(x1)
    durations_out = keras.layers.Dense(duration_units, activation='softmax', name='duration')(x1)
   
    model = keras.Model([note_input1, duration_input1], [notes_out, durations_out]) 
    
    model.compile(loss=LOSS,
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy", "accuracy"])

    model.summary()

    return model





build=True
if __name__ == "__main__":
    with open(os.path.join(SAVE_DIR, 'input'), 'rb') as f:
        inputs = pickle.load(f)
    with open(os.path.join(SAVE_DIR, 'output'), 'rb') as f:
        targets = pickle.load(f)
    with open(os.path.join(SAVE_DIR, 'length'), 'rb') as f:
        length = pickle.load(f)

    n_note=length[0]+1
    n_duration=length[1]+1
    print("hello world")
    print(inputs[0].shape)
    print(targets[0].shape)
    print(inputs[1].shape)
    print(targets[1].shape)
    
    if build:
        
        model = build_model(n_note,n_duration, rnn_units,  LEARNING_RATE)
        model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)
        model.save(SAVE_MODEL_PATH)

    else:
        model=keras.models.load_model(SAVE_MODEL_PATH)
        model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

        model.save(SAVE_MODEL_PATH)

    
