import os
import re
import codecs
import lib.words as wd
import lib.article_numbers as an

def sentences_doc_statuto(doc, rep=r''):
    txt = wd.clean_string_not_compressed(doc)
    clean_txt = wd.replace_evil_dots(txt, rep=rep)
    temp = wd.split_sents_newline(clean_txt) 
    res = [x.replace("<DOT>",".") for x in temp]
    #res = [wd.compress_blanks(x.replace("<DOT>",".")) for x in temp]
    return res

# Funzione di production, da copiare in article_numbers.py
# Prende in ingresso il testo passato solo da Textract e restituisce una stringa con tutto lo statuto
def extract_statuto(text):
    begin_statuto = ['S T A T U T O', 'STATUTO', 'STATUTO DELLA SOCITET', 'FUNZIONAMENTO DELLA SOCIET', 'PATTI DISCIPLINANTI']
    its_a_trap = ["ATTO COSTITUTIVO", "COSTITUZIONE DI SOCIET", "si allega", "lettera"]
    
    inizio = -1
    index = -1
    fine = -1
    
    # Just to be sure
    text = text.strip()

    sents = sentences_doc_statuto(text, rep=r'<DOT>')  
    for k, frase in enumerate(sents):
        found = False
        if index == -1:
            for t in its_a_trap:
                if wd.index_ignore_whitespaces(frase, t) > -1:
                    found = True
                    break
            if found:
                continue

            for b in begin_statuto:
                j = wd.index_ignore_whitespaces(frase, b)
                if j > -1:
                    inizio = k
                    index = j
                    break

    if index > -1:     
        # end_statuto_text restituisce la prima frase dopo la fine dello statuto
        res = an.end_statuto_init_text(wd.clean_string_not_compressed(text), sents[inizio][index:])
        if res != '':
            res = res + '\n'
            for y, frase2 in reversed(list(enumerate(sents))):
                if res in frase2:
                    fine = y
                    index_fine = frase2.index(res)
                    break

        for z, fr in enumerate(sents):
            if "RELAZIONE DI STIMA" in fr:
                fine = z
                index_fine = fr.index("RELAZIONE DI STIMA")
                break

    if index > -1 and fine > -1:
        to_write = [wd.compress_blanks(s.strip()) for s in sents[inizio+1:fine]]
        return([wd.compress_blanks(sents[inizio][index:])] + to_write + [wd.compress_blanks(sents[fine][:index_fine])])
    elif index > -1 and fine == -1:
        to_write = [wd.compress_blanks(s.strip()) for s in sents[inizio+1:]]
        return([wd.compress_blanks(sents[inizio][index:])] + to_write)
    else:
        # Statuto non trovato
        return []
