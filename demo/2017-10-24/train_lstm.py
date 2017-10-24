# This Python file uses the following encoding: utf-8
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

from lib.pretty_testing import predict_test

import numpy as np
import pickle

# fix random seed for reproducibility
np.random.seed(8)


# load prepared data
with open('embedded_docs.p') as f:
    data, labels = pickle.load(f)

print 'Data loaded'
    
# padding for the rnn
padded_data = sequence.pad_sequences(data, maxlen=200,padding="pre", truncating="post", value=0.0, dtype='float32')

print 'Data padded'

# load the dataset but only keep the top n words, zero the rest
X_train, X_test, y_train, y_test = train_test_split(padded_data, labels, train_size=0.9, stratify=labels)

print 'Data splitted'

model = Sequential()
model.add(LSTM(100, input_shape = (200, 100)))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

print 'Start training'
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize weights to HDF5
model.save_weights("models/keras_new_weights.h5")

# test
predict_test(model, X_test, y_test, ['non_cost', 'cost'])