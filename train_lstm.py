# This Python file uses the following encoding: utf-8
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json
import signal
from keras.optimizers import Adam, RMSprop
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from lib.pretty_testing import predict_test

import numpy as np
import pickle

# to catch ctrl C
#signal.signal(signal.SIGINT, signal.default_int_handler)

# fix random seed for reproducibility
np.random.seed(8)

dataset_file = 'embedded_docs_with_verb.p'
model_weights_file = 'models/keras_verb_checkpoint.h5' #'models/keras_deep_with_verb_es.h5'
model_file = 'models/keras_model_verb.json' #'models/keras_model.json'
lr = 0.0001
epochs = 100
training = False

# load prepared data
with open(dataset_file) as f:
    data, labels = pickle.load(f)

print 'Data loaded'
    
# padding for the rnn
padded_data = sequence.pad_sequences(data, maxlen=200,padding="pre", truncating="post", value=0.0, dtype='float32')
del data

print 'Data padded'

# load the dataset but only keep the top n words, zero the rest
X_train, X_test, y_train, y_test = train_test_split(padded_data, labels, train_size=0.9, stratify=labels)
del padded_data

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.95, stratify=y_train)

print 'Data splitted'

'''
model = Sequential()
model.add(LSTM(100, input_shape = (200, 100), return_sequences=True))
#model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=lr), metrics=['accuracy'])
print(model.summary())
'''

if training:
    model = Sequential()
    model.add(LSTM(100, input_shape = (200, 100)))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=lr), metrics=['accuracy'])
    print(model.summary())

    with open(model_file,'w') as f:
        f.write(model.to_json())
    
    print 'Start training'
    #try:
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_val, y_val),
              callbacks=[EarlyStopping(monitor='val_acc', patience=3),
                         ModelCheckpoint(model_weights_file, monitor='val_acc', save_best_only=True, save_weights_only=True)])
    #except KeyboardInterrupt:
    #    print('Captured ctrl-c, reloading checkpointed weights')
    #    model.load_weights(model_weights_file)
else:
    model = model_from_json(open(model_file).read())
    model.load_weights(model_weights_file)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=lr), metrics=['accuracy'])
    print(model.summary())
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize weights to HDF5
model.save_weights(model_weights_file)

# test
predict_test(model, X_test, y_test, ['non_cost', 'cost'])