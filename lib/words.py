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
allowed_lower_chars_punct = set(u'abcdefghijklmnopqrstuvwxyz1234567890àèéùòì!#$%&()*"+,-./:;<=>?@[\\]^_\'`{|}~ \t\n\r')

def to_utf8(txt):
    """
    Decodes UTF-8 strings to unicode objects
    """
    return txt.decode('utf-8') if type(txt) == str else txt

def compress_blanks(s):
    """
    Replaces all blank chars with single spaces
    """
    return re.sub( ur'\s+', u' ', s).strip()

def clean_string(s):
    """
    Converts to UTF-8, eliminates not meaningful chars (wrt the italian language), and compresses blanks
    """
    us = to_utf8(s)
    return compress_blanks(u''.join(c if c.lower() in allowed_lower_chars_punct else ' ' for c in us))

def clean_string_not_compressed(s):
    """
    Converts to UTF-8, eliminates not meaningful chars (wrt the italian language)
    """
    us = to_utf8(s)
    return u''.join(c if c.lower() in allowed_lower_chars_punct else ' ' for c in us)

def splitted_words(txt):
    """
    Basic, not UTF-8 way to split words (sequences of english letters and numbers)
    """
    return re.sub('[^\w]',' ',txt).split()

def splitted_words_utf8(txt):
    """
    Split words, keeping also accents, etc
    """
    return re.sub(ur'[^\w]',' ',txt, flags=re.UNICODE).split()

def replace_num(word, num_repl=u'NUM', max_digits=1):
    """
    Replaces a numeric word with NUM. By default single digit numbers are kept
    """
    return num_repl if word.isnumeric() and len(word)>max_digits else word

"""-----------------------OLD STUFF----------------------"""
def replace_num_old(word, num_repl=u'NUM'):
    return num_repl if num_repl in word else word

def replace_digit(c, r='NUM'):
    return r if c.isdigit() else c

def replace_digits(txt, r = 'NUM'):
    return ''.join(replace_digit(d, r) for d in txt)
"""-----------------------------------------------------"""

def replace_alnum(word, alnum_repl=u'ALPHANUM', max_digits=1):
    """
    Replaces strings containing digits with ALPHANUM
    """
    return alnum_repl if re.search('\d', word) != None and len(word)>max_digits else word

def contains_punctuation(word):
    sw = set(word)
    return len(sw-punctuation)<len(sw)

def contains_not_allowed_chars(word):
    swl = set(word.lower())
    return len(swl-allowed_lower_chars)>0

def split_sents(doc):
    """
    Basic way to split sentences with punctuation
    """
    return [s for s in re.split(u'[.;!?]', doc) if len(s)>0]

def replace_evil_dots_and_underscores(txt, rep=r''):
    """
    Eliminates dots in abbreviations and numbers, in order to split sentences on dots (that really divide sentences).
    It also removes underscores.
    """
    no_abbr = re.sub(u'([bcdfghjklmnpqrstvwxyz])\.', r'\1'+rep, txt)
    no_nums = re.sub(u'(\d)\.', r'\1'+rep, no_abbr)
    no_nums = re.sub(u'\.(\d)', r'\1'+rep, no_nums)
    no_maiusc = re.sub(u'([A-Z])\.', r'\1'+rep, no_nums)
    no_underscores = re.sub(u'_', ''+rep, no_maiusc)
    return no_underscores

def replace_evil_dots_and_underscores_newline(txt, rep=r''):
    """
    Eliminates dots in abbreviations and numbers, in order to split sentences on dots (that really divide sentences).
    It handles the case in which those dots separated sentences (when they are followed by a newline)
    It also removes underscores.
    """
    no_abbr = re.sub(r'([bcdfghjklmnpqrstvwxyz])\.(?!\n)', r'\1'+rep, txt)
    no_nums = re.sub(r'(\d)\.(?!\n)', r'\1'+rep, no_abbr)
    no_nums = re.sub(r'\.(\d)', r'\1'+rep, no_nums)
    no_maiusc = re.sub(r'([A-Z])\.(?!\n)', r'\1'+rep, no_nums)
    no_underscores = re.sub(r'_', ''+rep, no_maiusc)
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

def sentences_doc(doc, rep=r'', newline=False):
    if newline:
        txt = clean_string_not_compressed(doc)
        clean_txt = replace_evil_dots_and_underscores_newline(txt, rep=rep)
        clean_txt = compress_blanks(clean_txt)
    else:
        txt = clean_string(doc)
        clean_txt = replace_evil_dots_and_underscores(txt, rep=rep)
    return split_sents(clean_txt) 

def tokenize_sentences(sents, min_words=2, replace_nums=True, rm_stop_words=True):
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

def tokenize_doc(doc, min_words=2, replace_nums=True, rm_stop_words=True, rep=r'', newline=False):
    sents = sentences_doc(doc, rep, newline)
    return tokenize_sentences(sents, min_words, replace_nums, rm_stop_words)

def filter_indexes(pred, l):
    fl = ((i,e) for i,e in enumerate(l) if pred(e))
    return tuple(list(e) for e in zip(*fl))

def index_ignore_whitespaces(s, sub):
    ws = [' ', '\t', '\n', '\r', '\f', '\v']
    pred = lambda x: x not in ws
    idxs, fs = filter_indexes(pred, s)
    try:
        i = ''.join(fs).index(re.sub(r'\s+', '', sub))
    except:
        return -1
    return idxs[i]

def word_tokenize_replace(text, maxtokens=-1):
    """
    Tokenizes separating words and punctuation.
    It does the following substitutions:
        - italian stopwords with STOPWORD
        - dots with DOT
        - other punctuation with PUNCT
        - numbers with NUM
        - words with numbers (ex codes) with ALPHANUM
    """
    t = WordPunctTokenizer()
    cleaned_string = clean_string_not_compressed(text)
    clean_text = replace_evil_dots_and_underscores_newline(cleaned_string, rep=' ').lower()
    tokens = t.tokenize(clean_text)
    if maxtokens>0:
        tokens = tokens[:maxtokens]
    sw_tokens = (w if w not in stop_words else u"STOPWORD" for w in tokens)
    no_dots_tokens = (w if w != u'.' else u'DOT' for w in sw_tokens)
    no_punct_tokens = (w if w[0] not in punctuation else u'PUNCT' for w in no_dots_tokens)
    replaced_codes = [replace_alnum(replace_num(t)) for t in no_punct_tokens]
    return replaced_codes


def read_codec_file(filename, encoding='utf-8'):
    return codecs.open(filename, encoding=encoding).read()

'''
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
'''

#copy_pdfs_with_few_words2("out_txt", "res", "scans", "txt_scans", lambda f: os.path.getsize(os.path.abspath(f))<(1024))

#copy_pdfs_with_few_words("out_txt", "res", "scans", "txt_scans", 75)