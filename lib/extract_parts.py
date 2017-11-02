from __future__ import print_function
from __future__ import division
from keras.models import model_from_json
from keras.preprocessing import sequence
import predict_pdf as pp
import words as wd
import embedding as em
import extract_statuto as es
import text_extraction as te
import pandas as pd
import numpy as np
import json
from utils import uniq_list
from collections import OrderedDict
import re

default_parts = {k:[] for k in ['poteri', 'assemblea', 'clausola', 'non_riconducibile', 'scadenza']}

def join_cognomi_articles(words):
    articles = set([u'de', u'di', u'du', u'del', u'dell', u'la', u'dei', u'le', 
                    u'della', u'dall', u'dalla', u'dello', 'degli', 'lo', 'art',
                   u'dal', u'dalle'])
    found = False
    joined_words = []
    for word in words:
        if not found and word.lower() in articles:
            found = True
            cur_word = word
        elif found:
            cur_word += ' '+word
            joined_words.append(cur_word)
            found = False
        else:
            joined_words.append(word)
    return joined_words

def is_scadenza(s):
    return re.match(r'.*primo\s+?esercizio.*', s) != None

def post_process_prediction(sents, y_pred, neutral, milen=10):
    y_post = list(y_pred)
    for i in range(1,len(y_pred)-1):
        y_post[i] = y_post[i+1] if y_post[i] == neutral and y_post[i-1] == y_post[i+1] and len(sents[i])>=milen else y_post[i]
    return y_post

class PartsExtraction(object):
    def __init__(self, keras_model, reduced_dict, labels, maxlen = 100):
        self._model = keras_model
        self._reduced_dict = reduced_dict
        self._labels = labels
        self._maxlen = maxlen
        
    @staticmethod
    def load_from_files(keras_model_filename, keras_weights_filename,
                        reduced_dict_filename, labels=['poteri', 'assemblea', 'clausola', 'non_riconducibile', 'scadenza']):
        with open(keras_model_filename) as f:
            km = model_from_json(f.read())
        km.load_weights(keras_weights_filename)
        with open(reduced_dict_filename) as f:
            rd = json.load(f)
        return PartsExtraction(km, rd, labels)
    
    def _int_sentences(self, sentences):
        splitted_sentences = wd.tokenize_sentences(sentences, min_words=1)
        permitted_words = self._reduced_dict.keys()
        reduced_sentences = list(em.reduce_dictionary(splitted_sentences, permitted_words, min_words=1))
        return [[self._reduced_dict[w] for w in sent] for sent in reduced_sentences]
    
    def extract_parts_prob(self, sentences):
        int_sentences = self._int_sentences(sentences)
        padded_data = sequence.pad_sequences(int_sentences, self._maxlen, padding="pre", truncating="post", value=0, dtype='uint32')
        return self._model.predict(padded_data)
    
    def extract_parts(self, sentences, post_process=False, probas = np.array([])):
        if len(sentences) == 0:
            return []
        if len(probas)==0:
            probas = self.extract_parts_prob(sentences) 
        predictions = probas.argmax(axis=-1)
        for i in range(len(predictions)):
            if is_scadenza(sentences[i]):
                predictions[i] = self._labels.index('scadenza')
                break
        if post_process:
            preds = post_process_prediction(sentences, predictions, self._labels.index('non_riconducibile'))
        else:
            preds = predictions
        return [self._labels[i] for i in preds]
    
    def extract_parts_dict(self, sentences, predictions=None):
        predictions = self.extract_parts(sentences) if predictions == None else predictions
        df = pd.DataFrame({'sentence':sentences,'prediction':predictions})
        pivoted = df.pivot(columns='prediction', values='sentence')
        found_labels = set(predictions)
        return {k:list(filter(None, pivoted[k])) for k in found_labels}
    
    def extract_parts_dict_indexes(self, predictions, default_parts=default_parts):
        df = pd.DataFrame({'sentence':list(range(len(predictions))),'prediction':predictions})
        pivoted = df.pivot(columns='prediction', values='sentence')
        found_labels = set(predictions)
        res = default_parts.copy()
        res.update({k:[int(i) for i in filter(lambda x: x==x, pivoted[k])] for k in found_labels}) #nan != nan 
        return res
    
def is_valid_nl(txt, threshold=0.075):
    return txt.count('\n')/len(txt)<=threshold

def labels_probas_dict(labels, p):
    return {l:pr for l,pr in zip(labels, p)}

def sentences_probas_dict(sentences, probas):
    return [{'frase':s,'prob':labels_probas_dict(labels[:-1], p)} for s,p in zip(sentences, probas)]

def sentences_probas(sentences, probas):
    return [{'frase':s,'prob':p.tolist()} for s,p in zip(sentences, probas)]

def dict_indexes_to_sentences(sentences, dict_indexes):
    return {k:[sentences[i] for i in dict_indexes[k]] for k in dict_indexes.keys()}

def join_apostrophe(words):
    joined = ' '.join(words)
    rep_d = joined.replace(' d ', ' d\'').replace(' D ', ' D\'')
    return rep_d.split()

def extract_notaio_sent(sentence, notaio_names):
    words = join_apostrophe(wd.splitted_words_utf8(sentence))
    words = join_cognomi_articles(words)
    return [word for word in words if word.lower() in notaio_names and word[0].isupper()]

def find_me(words):
    lwords = [w.lower() for w in words]
    im = -1
    arts = set(['me', 'sottoscritto', 'mir'])
    for i, word in enumerate(lwords):
        if any((w in arts) for w in word.split()):
            return i
    return im

class NotaioNameExtractor(object):
    def __init__(self, notaio_names):
        self.notaio_names = notaio_names
        
    @staticmethod
    def load_from_file(filename='../dictionaries/nomi_notai.txt'):
        nomi_cognomi_list = [w for n in open(filename) for w in n.strip().lower().split(' ')]
        nomi_cognomi_notai = set(join_cognomi_articles(nomi_cognomi_list))
        return NotaioNameExtractor(nomi_cognomi_notai)

    def extract_notaio_name(self, doc, neigh=5, max_names=3):
        words = [w for sent in doc for w in wd.splitted_words_utf8(sent)]
        j_words = join_cognomi_articles(join_apostrophe(words))
        im = find_me(j_words)
        if im >= 0:
            m_words = j_words[im+1:im+neigh]
        else:
            m_words = j_words
        found_names = [word for word in m_words if word.lower() in self.notaio_names and word[0].isupper()]
        return uniq_list(found_names)[:max_names]

#def default_parts(labels=['poteri', 'assemblea', 'clausola', 'non_riconducibile', 'scadenza']):
#    return {k:[] for k in labels}
    
def build_response_dict(prediction=0, sensato=False, sentences=[], statuto=[], probas=[],  nome_notaio='', parts=default_parts, exception=True):
    classes_names = ['non costitutivo', 'costitutivo']
    res = {}
    res['classe'] = classes_names[int(round(prediction))]
    res['confidenza'] = prediction if prediction>0.5 else 1-prediction
    #if res['classe'] == 'costitutivo':
    res['sensato'] = sensato
    #if sensato == True:
    res['frasi'] = sentences_probas(sentences, probas)
    res['statuto'] = statuto
    res['nome_notaio'] = nome_notaio
    res['parti'] = parts
    res['exception'] = exception
    return res

class PredictorExtractor(object):
    def __init__(self, predictor_fn, predictor_models, parts_extractor, name_extractor): #, word_embedding = False):
        self.predict = predictor_fn
        self.predictor_models = predictor_models
        self.parts_extractor = parts_extractor
        self.name_extractor = name_extractor
        
    def extract_parts_pdf(self, filename):
        txt = te.extract_text(filename, do_ocr=False, pages=-1)
        sentences = wd.sentences_doc(txt, rep=' ', newline=True)
        pe = self.parts_extractor
        probas = pe.extract_parts_prob(sentences)
        predictions = pe.extract_parts(sentences, post_process=True, probas=probas)
        return list(zip(sentences, predictions, *zip(*probas)))

    def predict_extract_pdf_dict(self, filename):
        txt = te.extract_text(filename, do_ocr=False, pages=-1)
        
        prediction = float(self.predict(txt, **self.predictor_models))
        sensato = is_valid_nl(txt)
        
        if prediction <0.5 or not sensato:
            return build_response_dict(prediction, exception=False)
        
        sentences = wd.sentences_doc(txt, rep=' ', newline=True)

        try:
            statuto = es.extract_statuto(txt)
        except Exception as e:
            print ('eccezione:', e)
            statuto = []
        
        pe = self.parts_extractor
        probas = pe.extract_parts_prob(sentences)
        predictions = pe.extract_parts(sentences, post_process=True, probas=probas)
        dict_indexes = pe.extract_parts_dict_indexes(predictions)

        name = ' '.join(self.name_extractor.extract_notaio_name(sentences))
        return build_response_dict(prediction, sensato, sentences, statuto, probas, name, dict_indexes, False)
    