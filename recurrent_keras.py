from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

import numpy as np
import pickle


# fix random seed for reproducibility
np.random.seed(86)

num_sentences = 200

# load prepared data
with open('embedded_docs.p') as f:
    data, labels = pickle.load(f)


# padding for the rnn
padded_data = sequence.pad_sequences(data, maxlen=num_sentences,padding="pre", truncating="post", value=0.0, dtype='float32')

# Split the dataset in train and test set
X_train, X_test, y_train, y_test = train_test_split(padded_data, labels, train_size=0.9, stratify=labels)


model = Sequential()
model.add(LSTM(100, input_shape = (num_sentences, 100)))
#model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# serialize weights to HDF5
model.save_weights("keras_weights.h5")
