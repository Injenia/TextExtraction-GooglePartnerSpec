# This Python file uses the following encoding: utf-8
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

import numpy as np
import re
import string
import os
import operator
import pickle

import random
import pandas as pd


# Creazione del dataset come sottoinsieme bilanciato dei documenti

csv_filename = '/notebooks/dev/infocamere/atti.csv'
model_filename = '/notebooks/dev/infocamere/git/models/gensim_model.d2v'

df = pd.read_csv(csv_filename)

size_nc = len(df.loc[df['label'] == 'non_costitutivo'].groupby('filename'))

grouped = df.loc[df['label'] == 'costitutivo'].groupby(df["filename"])
dfs = [g[1] for g in list(grouped)[:size_nc]]

grouped_nc = df.loc[df['label'] == 'non_costitutivo'].groupby(df["filename"])
dfs_nc = [g[1] for g in list(grouped_nc)]

df_balanced = pd.concat(dfs + dfs_nc)

pd_sentences = df_balanced['sentence']

print "DF created"


# Creazione degli embedding

def build_dictionary(sentences):
    d = dict()
    index = 0
    for sentence in sentences:
        for word in sentence:
            if not word in d:
                d[word] = index
                index += 1
    return d

def word_counts(sentences):
    d = dict()
    for sentence in sentences:
        for word in sentence:
            if not word in d:
                d[word] = 1
            else:
                d[word] += 1
    return d

def rev_dict(d):
    rd = dict()
    for w,i in d.items():
        rd[i] = w
    return rd


#Sentence iterator for building the gensim model

def iter_sentences(sents):
    i = 0
    for line in sents:
        yield LabeledSentence(line, ['SENT_%s' % i])
        i += 1
        

# Modello dell'embedding

def build_embedding(sentences, refresh=False, epochs = 10):
    if not refresh and os.path.exists(model_filename):
        model = Doc2Vec.load(model_filename)
    else:
        model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-5, negative=5, workers=2)
        model.build_vocab(sentences)
        print 'Vocabulary built'
        #for epoch in range(5):
        #    print 'Epoch', epoch
        model.train(sentences, model.corpus_count, epochs = epochs)
        model.save(model_filename)
        print 'Model saved'
    return model

def first_n_words(dictionary, n):
    rd = rev_dict(d)
    wc = word_counts(s.split() for s in pd_sentences)
    sorted_wc = sorted(wc.items(), key=operator.itemgetter(1))
    return set(reversed([x[0] for x in sorted_wc[-n:]]))

def substitute_word(word, permitted_words, unknown = 'UNK'):
    return word if word in permitted_words else unknown

def reduced_sentence(sentence, permitted_words):
    return [substitute_word(word, permitted_words) for word in sentence]

def reduce_dictionary(sentences, permitted_words, min_words=2):
    for sentence in sentences:
        new_sentence = reduced_sentence(sentence, permitted_words)
        if len(new_sentence) >= min_words:
            yield new_sentence
            
def sentence_vector(model, sentence, permitted_words):
    return model.infer_vector(reduced_sentence(sentence.split(' '), permitted_words))

d = build_dictionary(s.split() for s in pd_sentences)
first_10000_words = first_n_words(d, 10000)

#filtered_sentences = reduce_dictionary((s.split() for s in pd_sentences), first_10000_words)
#filtered_sentences_list = list(filtered_sentences)

#model = build_embedding(list(iter_sentences(filtered_sentences)))
model = build_embedding(None)

# Costruzione del dataset

def build_dataset(model, df, permitted_words):
    filename = ""
    docs = []
    labels = []
    curdoc = []                  # lista delle frasi del documento corrente
    for i in xrange(len(df)):
        row = df.iloc[i] 
        if filename == "":
            filename = row["filename"]
            labels.append(row["label"])
            
        embedding = sentence_vector(model, row['sentence'], permitted_words)
        if filename == row["filename"]:
            curdoc.append(embedding)
        else:
            print "%s with len: %d" % (filename, len(curdoc))
            docs.append(curdoc)
            curdoc = [embedding]
            labels.append(row["label"])
            filename = row['filename']
    if len(curdoc)>0:
        docs.append(curdoc)
    return docs, labels

docs, labels = build_dataset(model, df_balanced, first_10000_words)
label_map = {'costitutivo':1, 'non_costitutivo':0}
labels_n = [label_map[l] for l in labels]

with open("/notebooks/dev/infocamere/git/embedded_docs.p", "w") as fout:
    pickle.dump([docs, labels_n], fout)

