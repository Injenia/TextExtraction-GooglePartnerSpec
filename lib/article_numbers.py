# This Python file uses the following encoding: utf-8
import re
import roman
import operator
import bisect

def index_matches(regex, lines, conv_ints=True):
    matches = (re.findall(regex, l) for l in lines)
    filt_matches = [(i,int(m[0]) if conv_ints else m[0]) for i,m in enumerate(matches) if m and len(m[0])>0]
    return filt_matches

def changes_only_iter(l, key_fn = lambda x:x):
    il = iter(l)
    t = next(il)
    yield t
    for i in il:
        if key_fn(i)!=key_fn(t):
            t = i
            yield i

def filter_matches_errors(ml):
    if len(ml) < 2:
        return ml
    res = [ml[0]]
    for i in range(1, len(ml)-1):
        if (ml[i][1] == ml[i-1][1]+1) or (ml[i][1] == ml[i+1][1]-1) and ml[i][1]!=1:
            res.append(ml[i])
    res.append(ml[-1])
    return res

def matches_list(lines):
    num_regex = ur'(?u)^-*\s*(\d{1,2})[°\.\)](?!\d)'
    art_regex = r'(?i)^-*_*\s*art\D{0,10}(\d{1,2})(?!\d)'
    roman_art_regex = r'(?i)^-*\s*?art\w*\.?\s+((?:ix)?x?(?:iv)?v?i{0,3})\s?'
    roman_num_regex = r'(?i)^((?:ix)?x{0,3}(?:ix)?(?:iv)?v?i{0,3})(?:\)|\.)\s?'
    
    nm = index_matches(num_regex, lines)
    num_matches = list(changes_only_iter(nm, operator.itemgetter(1)))

    im = index_matches(art_regex, lines)

    imr_r = index_matches(roman_art_regex, lines, conv_ints=False)
    imr = [(i, roman.fromRoman(r.upper())) for i,r in imr_r]
    
    imr_r_n = index_matches(roman_num_regex, lines, conv_ints=False)
    imr_n = [(i, roman.fromRoman(r.upper())) for i,r in imr_r_n]

    matches = sorted(num_matches + im + imr + imr_n, key=operator.itemgetter(0))
    return matches #list(changes_only_iter(matches, operator.itemgetter(1))))
            
def end_statuto(lines, matches):
    found = False
    found_num = -1
    end_statuto_line = ''
    if len(matches) == 0:
        return ''
    im = iter(matches)
    _, t = next(im)
    for i, n in im:
        if found == True and n != t + 1 and n == found_num + 1:
            end_statuto_line = lines[i]
            found = False
        if n < t:
            #print(n)
            found = True
            found_num = t
        t = n
    return end_statuto_line

def end_statuto_text(text):
    lines = [l.strip() for l in text.split('\n') if len(l.strip())>0]
    matches = matches_list(lines)
    corrected_matches = filter_matches_errors(matches)
    return end_statuto(lines, corrected_matches)

def text_idx_lines_idx(idx, lines):
    prev_idx = 0
    cur_idx = 0
    i_lines = 0
    while cur_idx < idx and i_lines < len(lines):
        prev_idx = cur_idx
        cur_idx += len(lines[i_lines]) + 1
        i_lines += 1
    return i_lines

'''
def num_before(l, n):
    prev = 0
    for e in l:
        if e < n:
            prev = e
        else:
            return prev
'''        

def num_before(l, n):
    return l[bisect.bisect_left(l,n)-1]

def end_statuto_init(lines, matches, init_idx):
    sent_idx_before = num_before([i for i,e in matches], init_idx)
    for i,e in matches:
        if i == sent_idx_before:
            idx_to_find = e + 1
            break
    
    im = iter(matches)
    _, t = next(im)
    
    for i,e in matches:
        if e == idx_to_find and e != t + 1:
            return lines[i]
        t = e
    return ''

def end_statuto_init_text(text, init_statuto_str):
    lines = [l.strip() for l in text.split('\n') if len(l.strip())>0]
    matches = matches_list(lines)
    corrected_matches = filter_matches_errors(matches)
    
    i_text = text.index(init_statuto_str)
    il = text_idx_lines_idx(i_text, lines)
    
    return end_statuto_init(lines, corrected_matches, il)
