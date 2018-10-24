from music21 import *
import glob
import pickle
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

notes_to_parse = None

notes = []  # contains elements only

sequence_length = 100

rest = True

for file in glob.glob("midi/*.mid"):
    # file = "midi/Wtcii01a.mid"
    midi = converter.parse(file)
    print("Parsing %s" % file)

    try:  # file has instrument parts
        inst = instrument.partitionByInstrument(midi)
        print("Number of instrument parts: " + str(len(inst.parts)))
        notes_to_parse = inst.parts[0].recurse()
    except:  # file has notes in a flat structure
        notes_to_parse = midi.flat.notes
        highest_offset = midi.flat.highestOffset

    for element in notes_to_parse:

        if isinstance(element, note.Note):
            notes.append(str(element.pitch))

        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

        elif isinstance(element, note.Rest) and rest:
            notes.append("rest")


with open('data/notes', 'wb') as filepath:
    pickle.dump(notes, filepath)

# for note in notes:
#     print(note)


pitchnames = sorted(set(item for item in notes))
note_to_int = dict((notes, number) for number, notes in enumerate(pitchnames))
print("Dictionary size: %f" % len(note_to_int))

network_input = []
network_output = []

n_vocab = len(set(notes))

# create input sequences and the corresponding outputs
print("Create input sequences and the corresponding outputs")
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])

    # print("outside of for loop", i)

n_patterns = len(network_input)

# reshape the input into a format compatible with LSTM layers
print("Reshape the input into a format compatible with LSTM layers")
network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
# normalize input
print("Normalize input")
network_input = network_input / float(n_vocab)

network_output = np_utils.to_categorical(network_output)

# Creating model
print("Creating model")
model = Sequential()
model.add(LSTM(
    512,
    input_shape=(network_input.shape[1], network_input.shape[2]),
    return_sequences=True
))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Training model
print("Training model")
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(
    filepath,
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)
callbacks_list = [checkpoint]

model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)





