import re
import os
from shutil import copyfile

join_path = os.path.join

stop_words = open('/notebooks/dev/infocamere/git/stop_words.txt').read().split()

def splitted_words(txt):
    return re.sub('[^\w]',' ',txt).split()

def replace_digit(c, r = 'NUM'):
    if c.isdigit():
        return r
    else:
        return c

def replace_digits(txt, r = 'NUM'):
    return ''.join(replace_digit(d, r) for d in txt)

def replace_num(word, r = 'NUM'):
    if r in word:
        return r
    else:
        return word

def replace_evil_dots_and_underscores(txt):
    no_cons = re.sub('([bcdfghjklmnpqrstvwxyz])\.', r'\1', txt)
    no_nums = re.sub('(\d)\.', r'\1', no_cons)
    no_nums = re.sub('\.(\d)', r'\1', no_nums)
    no_maiusc = re.sub('([A-Z])\.', r'\1', no_nums)
    no_underscores = re.sub('_', '', no_maiusc)
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
    sentences_rep = (list(map(replace_num, s)) for s in sentences)
    return [rm_stop_words(s) for s in sentences_rep if len(rm_stop_words(s))>=min_words]

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