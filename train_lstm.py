# This Python file uses the following encoding: utf-8
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam, RMSprop
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from lib.pretty_testing import predict_test

import numpy as np
import pickle

# fix random seed for reproducibility
np.random.seed(8)

dataset_file = 'embedded_docs_with_verb.p'
model_weights_file = 'models/keras_new_weights_with_verb_es.h5'
lr = 0.0001
epochs = 100

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

model = Sequential()
model.add(LSTM(100, input_shape = (200, 100)))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=lr), metrics=['accuracy'])
print(model.summary())

print 'Start training'
model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_val, y_val),
          callbacks=[EarlyStopping(monitor='val_acc', patience=5)])

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize weights to HDF5
model.save_weights(model_weights_file)

# test
predict_test(model, X_test, y_test, ['non_cost', 'cost'])