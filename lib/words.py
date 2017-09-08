# This Python file uses the following encoding: utf-8
import re
import os
from shutil import copyfile
import nltk
from nltk.tokenize import WordPunctTokenizer
#tokenizer = nltk.data.load('tokenizers/punkt/italian.pickle')
from nltk.corpus import stopwords
import string
import codecs

punctuation = set(string.punctuation)
stop_words = set(stopwords.words('italian'))
word_tokenizer = WordPunctTokenizer()
join_path = os.path.join
allowed_lower_chars = set(u'abcdefghijklmnopqrstuvwxyz1234567890àèéùòì')
#stop_words = open('/notebooks/dev/infocamere/git/stop_words.txt').read().split()

def splitted_words(txt):
    return re.sub('[^\w]',' ',txt).split()

def replace_num(word, num_repl=u'NUM', max_digits=1):
    return num_repl if word.isnumeric() and len(word)>max_digits else word

def replace_num_old(word, num_repl=u'NUM'):
    return num_repl if num_repl in word else word

def replace_digit(c, r='NUM'):
    return r if c.isdigit() else c

def replace_digits(txt, r = 'NUM'):
    return ''.join(replace_digit(d, r) for d in txt)

def replace_alnum(word, alnum_repl=u'ALPHANUM', max_digits=1):
    return alnum_repl if re.search('\d', word) != None and len(word)>max_digits else word

def contains_punctuation(word):
    sw = set(word)
    return len(sw-punctuation)<len(sw)

def contains_not_allowed_chars(word):
    swl = set(word.lower())
    return len(swl-allowed_lower_chars)>0

def split_sents(doc):
    return [s for s in re.split(u'[.;!?]', doc) if len(s)>0]

'''
def replace_num(word, r = 'NUM'):
    if r in word:
        return r
    else:
        return word
'''

def replace_evil_dots_and_underscores(txt):
    no_abbr = re.sub(u'([bcdfghjklmnpqrstvwxyz])\.', r'\1', txt)
    no_nums = re.sub(u'(\d)\.', r'\1', no_abbr)
    no_nums = re.sub(u'\.(\d)', r'\1', no_nums)
    no_maiusc = re.sub(u'([A-Z])\.', r'\1', no_nums)
    no_underscores = re.sub(u'_', '', no_maiusc)
    return no_underscores

def rm_stop_words(words):
    return [word for word in words if not (word in stop_words)]

def rm_underscores(words):
    return [word for word in words if not ('_' in word)]

def naive_split(txt, digit_replacement='NUM', split='.\n', min_words = 5):
    return [s for s in map(splitted_words, replace_digits(txt.lower(),digit_replacement).split(split)) if len(s)>=min_words]

def not_so_naive_split(txt, digit_replacement='NUM', split='.', min_words = 2):
    splitted = replace_digits(replace_evil_dots_and_underscores(txt).lower(),digit_replacement).split(split)
    sentences = map(splitted_words, splitted)
    sentences_rep = (list(map(replace_num_old, s)) for s in sentences)
    return [rm_stop_words(s) for s in sentences_rep if len(rm_stop_words(s))>=min_words]

def tokenize_doc(doc, min_words=2, replace_nums=True, rm_stop_words=True):
    if type(doc) == str:
        txt = doc.decode('utf-8')
    elif type(doc) == unicode:
        txt = doc
    clean_txt = replace_evil_dots_and_underscores(txt)
    sents = split_sents(clean_txt) #tokenizer.sentences_from_text(clean_txt)
    sents_words = word_tokenizer.tokenize_sents(sents)
    splitted_sents = []
    for sentence in sents_words:
        if replace_nums:
            sent = [replace_alnum(replace_num(word.lower())) for word in sentence if not contains_not_allowed_chars(word.lower())
                                                                                     and word not in stop_words]
        else:
            sent = [word.lower() for word in sentence if not contains_not_allowed_chars(word.lower()) and word not in stop_words]
        if len(sent)>=min_words:
            splitted_sents.append(sent)
    return splitted_sents

def read_codec_file(filename, encoding='utf-8'):
    return codecs.open(filename, encoding=encoding).read()

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
            
def copy_pdfs_with_few_words(txt_folder, pdf_folder, out_folder, txt_out_folder, thres):
    for txt_file in os.listdir(txt_folder):
        txt = open(join_path(txt_folder,txt_file)).read()
        n_words = len(splitted_words(txt))
        if n_words < thres:
            print '%s has %d words' % (txt_file, n_words)
            pdf_file_name = txt_file[:-3] + 'pdf'
            copyfile(join_path(pdf_folder,pdf_file_name),join_path(out_folder,pdf_file_name))
            os.rename(join_path(txt_folder,txt_file), join_path(txt_out_folder,txt_file))
            
def copy_pdfs_with_few_words2(txt_folder, pdf_folder, out_folder, txt_out_folder, pred):
    for f in os.listdir(txt_folder):
        if pred(join_path(txt_folder,f)):
            print f
            pdf_file_name = f[:-3] + 'pdf'
            copyfile(join_path(pdf_folder,pdf_file_name),join_path(out_folder,pdf_file_name))
            os.rename(join_path(txt_folder,f), join_path(txt_out_folder,f))


#copy_pdfs_with_few_words2("out_txt", "res", "scans", "txt_scans", lambda f: os.path.getsize(os.path.abspath(f))<(1024))

#copy_pdfs_with_few_words("out_txt", "res", "scans", "txt_scans", 75)