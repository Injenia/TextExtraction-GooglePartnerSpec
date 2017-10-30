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
    res = [ml[0]]
    for i in range(1, len(ml)-1):
        if (ml[i][1] == ml[i-1][1]+1) or (ml[i][1] == ml[i+1][1]-1):
            res.append(ml[i])
    res.append(ml[-1])
    return res

def matches_list(lines):
    num_regex = ur'(?u)^-*\s*(\d{1,2})[°\.\)](?!\d)'
    art_regex = r'(?i)^(?:- )?art\D{0,10}(\d{1,2})'
    roman_art_regex = r'(?i)^-*\s*?art\w*\.?\s+((?:ix)?(?:iv)?x?v?i{0,3})\s?'

    nm = index_matches(num_regex, lines)
    num_matches = list(changes_only_iter(nm, operator.itemgetter(1)))

    im = index_matches(art_regex, lines)

    imr_r = index_matches(roman_art_regex, lines, conv_ints=False)
    #print(imr_r)
    imr = [(i, roman.fromRoman(r)) for i,r in imr_r]

    matches = sorted(num_matches+im+ imr, key=operator.itemgetter(0))
    return matches
            
def end_statuto(lines, matches):
    found = False
    end_statuto_line = ''
    if len(matches) == 0:
        return ''
    im = iter(matches)
    _, t = next(im)
    for i, n in im:
        if found == True and n < t:
            end_statuto_line = lines[i]
            found = False
        if n < t:
            #print(n)
            found = True
        t = n
    return end_statuto_line

def end_statuto_text(text):
    lines = [l.strip() for l in text.split('\n') if len(l.strip())>0]
    matches = matches_list(lines)
    corrected_matches = filter_matches_errors(matches)
    return end_statuto(lines, corrected_matches)
