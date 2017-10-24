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
import json
import random
import pandas as pd

# Creazione degli embedding

def build_dictionary(sentences, start_index=0):
    d = dict()
    index = start_index
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


#Sentence iterator to build the gensim model

def iter_sentences(sents):
    i = 0
    for line in sents:
        yield LabeledSentence(line, ['SENT_%s' % i])
        i += 1
        

# Modello dell'embedding

def build_embedding(sentences, model_filename, refresh=False, epochs = 10):
    if not refresh and os.path.exists(model_filename):
        model = Doc2Vec.load(model_filename)
    else:
        model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-5, negative=5, workers=4)
        model.build_vocab(sentences)
        print 'Vocabulary built'
        model.train(sentences, model.corpus_count, epochs = epochs)
        model.save(model_filename)
        print 'Model saved'
    return model


def first_n_words(sentences, n):
    wc = word_counts(sentences)
    sorted_wc = sorted(wc.items(), key=operator.itemgetter(1))
    return list(reversed([x for x in sorted_wc[-n:]]))


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
    if type(sentence) == str:
        sent_list = sentence.split()
    else:
        sent_list = sentence
    return model.infer_vector(reduced_sentence(sent_list, permitted_words)) 

def embed_document(model, doc, permitted_words):
    return [sentence_vector(model, sentence, permitted_words) for sentence in doc]

#parallelizable...
def embed_document_p(doc, model, permitted_words):
    return [sentence_vector(model, sentence, permitted_words) for sentence in doc]

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

if __name__ == '__main__':
    csv_filename = '/notebooks/dev/infocamere/atti2.csv'
    model_filename = '../models/gensim_5000_model_with_verb.d2v'
    permitted_words_filename = '../first_5000_words_with_verb_cost.json'
    dataset_filename = "../embedded_docs_with_verb.p"
    
    # Creazione del dataset come sottoinsieme bilanciato dei documenti
    df = pd.read_csv(csv_filename, encoding='utf-8')

    size_nc = len(df.loc[df['label'] == 'non_costitutivo'].groupby('filename'))

    grouped = df.loc[df['label'] == 'costitutivo'].groupby(df["filename"])
    dfs = [g[1] for g in list(grouped)[:size_nc]]

    grouped_nc = df.loc[df['label'] == 'non_costitutivo'].groupby(df["filename"])
    dfs_nc = [g[1] for g in list(grouped_nc)]

    df_balanced = pd.concat(dfs + dfs_nc)

    pd_sentences = df_balanced['sentence']

    print "DF created"
    splitted_sentences = [s.split() for s in pd_sentences]
    
    if os.path.exists(permitted_words_filename):
        with open(permitted_words_filename) as o:
            permitted_words = json.load(o)
    else:
        fnw = first_n_words(splitted_sentences, 5000)
        permitted_words = [e[0] for e in fnw]
        with open(permitted_words_filename, 'w') as o:
            json.dump(permitted_words, o)
    
    filtered_sentences = reduce_dictionary(splitted_sentences, set(permitted_words))
    
    print("Freeing memory")
    del df
    del grouped
    del dfs
    del grouped_nc

    model = build_embedding(list(iter_sentences(filtered_sentences)), model_filename)
    #model = build_embedding(None)
    del splitted_sentences
    
    docs, labels = build_dataset(model, df_balanced, permitted_words)
    label_map = {'costitutivo':1, 'non_costitutivo':0}
    labels_n = [label_map[l] for l in labels]

    with open(dataset_filename, "w") as fout:
        pickle.dump([docs, labels_n], fout)

