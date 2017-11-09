# This Python file uses the following encoding: utf-8
from __future__ import absolute_import

from lib.embedding import embed_document, reduce_dictionary
import lib.embedding as em
from keras.preprocessing import sequence
from keras.models import model_from_json
from collections import OrderedDict
from gensim.models import Doc2Vec
from lib.words import tokenize_doc, word_tokenize_replace
from functools import partial
import lib.text_extraction as te
import pandas as pd
import json

def predict_documents_str(filenames, txts, gensim_model, keras_model, permitted_words):
    filtered_filenames = [f for f,t in zip(filenames, txts) if (t != None and len(t)>0)]
    not_empty_txts =  [t for t in txts if  (t != None and len(t)>0)]
    if len(filtered_filenames) == 0:
        return [], []

    splitted_txts = (tokenize_doc(txt) for txt in not_empty_txts)
    embedded_txts = [embed_document(gensim_model, doc, permitted_words) for doc in splitted_txts] #filtered_txts]
    padded_data = sequence.pad_sequences(embedded_txts, maxlen=200, padding="pre", truncating="post", value=0.0, dtype='float32')
    probs = keras_model.predict_proba(padded_data, verbose=0)
    return [prob[0] for prob in probs], filtered_filenames

def predict_documents_pdf(filenames, gensim_model, keras_model, permitted_words, do_ocr=False):
    txts = [te.extract_text(filename, do_ocr) for filename in filenames]
    return predict_documents_str(filenames, txts, gensim_model, keras_model, permitted_words)

def predict_documents_str_we(filenames, txts, keras_model, reduced_dictionary, maxlen=500):
    filtered_filenames = [f for f,t in zip(filenames, txts) if (t != None and len(t)>0)]
    not_empty_txts =  [t for t in txts if  (t != None and len(t)>0)]
    if len(filtered_filenames) == 0:
        return [], []

    tokenized_txts = (word_tokenize_replace(txt, maxlen) for txt in not_empty_txts)
    d = em.DictionaryMapper(reduced_dictionary)
    
    int_txts = list(d.map_to_ints(tokenized_txts))
    padded_data = sequence.pad_sequences(int_txts, maxlen, padding="pre", truncating="post", value=0, dtype='uint32')
    probs = keras_model.predict_proba(padded_data, verbose=0)
    return [prob[0] for prob in probs], filtered_filenames

def predict_documents_pdf_we(filenames, keras_model, reduced_dictionary, do_ocr=False):
    txts = [te.extract_text(filename, do_ocr) for filename in filenames]
    return predict_documents_str_we(filenames, txts, keras_model, reduced_dictionary)


def predict_documents_txt(filenames, gensim_model, keras_model, permitted_words):
    txts = [open(filename).read() for filename in filenames]
    return predict_documents_str(filenames, txts, gensim_model, keras_model, permitted_words)

def load_models(gensim_file='../models/gensim_model_5000.d2v', 
                keras_model_file='../models/keras_model.json',
                keras_weights_file='../models/keras_weights_5000.h5',
                permitted_words_file='../dictionaries/first_5000_words.json'):
    models = {}
    models['gensim_model'] = Doc2Vec.load(gensim_file)

    with open(keras_model_file) as f:
        models['keras_model'] = model_from_json(f.read())
    models['keras_model'].load_weights(keras_weights_file)

    with open(permitted_words_file) as f:
        models['permitted_words'] = set(json.load(f))
    return models

def load_models_we(keras_model_file, keras_weights_file, reduced_dictionary_file):
    models = {}
    with open(keras_model_file) as f:
        models['keras_model'] = model_from_json(f.read())
    models['keras_model'].load_weights(keras_weights_file)

    with open(reduced_dictionary_file) as f:
        models['reduced_dictionary'] = json.load(f)
    return models

def predict_document_str(txt, gensim_model, keras_model, permitted_words):
    split_txt = tokenize_doc(txt)
    embedded_txt = embed_document(gensim_model, split_txt, permitted_words)
    padded_data = sequence.pad_sequences([embedded_txt], maxlen=200, padding="pre", truncating="post", value=0.0, dtype='float32')
    return keras_model.predict(padded_data, verbose=0)[0,0]

def predict_document_str_we(txt, keras_model, reduced_dictionary, maxlen = 500):
    tokenized_txt = word_tokenize_replace(txt, maxlen)
    d = em.DictionaryMapper(reduced_dictionary)
    int_txt = next(d.map_to_ints([tokenized_txt]))
    padded_data = sequence.pad_sequences([int_txt], maxlen, padding="pre", truncating="post", value=0, dtype='uint32')
    return keras_model.predict(padded_data, verbose=0)[0,0]

def load_prediction_models(gensim_file='../models/gensim_model_5000.d2v', 
                           keras_model_file='../models/keras_model.json',
                           keras_weights_file='../models/keras_weights_5000.h5',
                           permitted_words_file='../dictionaries/first_5000_words.json'):
    return partial(predict_documents_pdf, **load_models(gensim_file, keras_model_file, keras_weights_file, permitted_words_file))
    
def predictions_dataframe(pdf_names, filtered_filenames, predictions, csv_out_file, labels_map=['NON COSTITUTIVO', 'COSTITUTIVO']):
    filt_filenames_set = set(filtered_filenames)
    labels = [labels_map[int(round(pred))] for pred in predictions]
    not_predicted_pdfs = [pdf for pdf in pdf_names if pdf not in filt_filenames_set]
    err_fill = [u'']*len(not_predicted_pdfs)
    pred_fill = [u'']*len(filtered_filenames)
    err_msgs = [u'scansione']*len(not_predicted_pdfs)
    
    df_dict = OrderedDict([('Nome file', filtered_filenames+not_predicted_pdfs),
                          ('Errore', ['No']*len(filtered_filenames) + ['Si']*len(not_predicted_pdfs)),
                          ('Messaggio errore', pred_fill + err_msgs),
                          ('Output rete', predictions + err_fill),
                          ('Predizione', labels + err_fill),])
    
    df = pd.DataFrame(df_dict)
    return df
