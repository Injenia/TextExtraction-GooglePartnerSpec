# This Python file uses the following encoding: utf-8
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, Embedding
from keras.models import model_from_json
import signal
from keras.optimizers import Adam, RMSprop
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
from lib.pretty_testing import predict_test #, roc_curve_plot
from sklearn.metrics import roc_curve, auc

import numpy as np
import pickle

# to catch ctrl C
#signal.signal(signal.SIGINT, signal.default_int_handler)

# fix random seed for reproducibility
np.random.seed(8)

dataset_file = '../datasets/word_embedded_docs.p'
model_weights_file = '../models/keras_weights_word_embedding.h5'
model_file = '../models/keras_model_word_embedding.json'
roc_fig_filename = '../log_figs/roc_curve.png'
lr = 0.0003
epochs = 100
training = False
patience = 3
top_words = 10000
embedding_vector_length = 32
maxlen = 500

# load prepared data
with open(dataset_file) as f:
    data, labels = pickle.load(f)

print 'Data loaded'
    
# padding for the rnn
padded_data = sequence.pad_sequences(data, maxlen, padding="pre", truncating="post", value=0, dtype='uint32')
labels = np.array(labels)
del data

print 'Data padded'

# load the dataset but only keep the top n words, zero the rest
X_train, X_test, y_train, y_test = train_test_split(padded_data, labels, train_size=0.9, stratify=labels)
del padded_data

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.95, stratify=y_train)

print 'Data splitted'


if training:
    model = Sequential()
    model.add(Embedding(top_words, embedding_vector_length, input_length=maxlen, mask_zero=True))
    model.add(LSTM(32, dropout=0.2)) #return_sequences=True
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=lr), metrics=['accuracy'])
    print(model.summary())
    
    with open(model_file,'w') as f:
        f.write(model.to_json())
    
    print 'Start training'

    model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_data=(X_val, y_val),
              callbacks=[EarlyStopping(monitor='val_acc', patience=patience),
                         ModelCheckpoint(model_weights_file, monitor='val_acc', save_best_only=True, save_weights_only=True)])
else:
    model = model_from_json(open(model_file).read())
    model.load_weights(model_weights_file)
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=lr), metrics=['accuracy'])
    print(model.summary())
    
# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize weights to HDF5 (there's the checkpoint)
#model.save_weights(model_weights_file)

# test
predict_test(model, X_test, y_test, ['non_cost', 'cost'])
scores = model.predict(X_test, verbose=0).reshape(-1)

fpr, tpr, _ = roc_curve(y_test, scores, pos_label=1)
roc_auc = auc(fpr, tpr)
print('\nArea under ROC curve: {}'.format(roc_auc))
#fig = roc_curve_plot(fpr, tpr, roc_auc)
#fig.savefig(roc_fig_filename)
