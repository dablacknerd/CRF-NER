import re
from custom_nlp_functions import *
from resources import *


def preprocess_token(token):
    symbols =['*','.','-',';',':',"'",'"','~','=']
    symbols_regex = re.compile(r'[\.|-]+')
    if len(token) <= 3:
        return 'NDA'
    elif re.match(symbols_regex,token):
        return 'NDA'
    for symbol in symbols:
        token_len = len(token)
        start= 0
        end = token_len - 1
        if token[start] == symbol:
            token = token.lstrip(symbol)
            token_len = len(token)
            end = token_len - 1
        try: 
            if token[end] == symbol:
                token = token.rstrip(symbol)
        except IndexError:
            return 'NDA'
    return token

def is_a_date(token):
    #print(token)
    new_token = preprocess_token(token)
    date_regex = re.compile(r"\s*\d{1,2}[\/|.|-]\d{1,2}[\/|.|-]\d{2,4}|\d{4}|\d{1,2}[\/|.|-]\d{2,4}\s*")
    if re.match(date_regex,new_token):
        return True
    else:
        return False

def word2features(sent, i):
    word = str(sent[i][0])
    postag = sent[i][1]
    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'is_date':is_a_date(word),
    }
    if i > 0:
        word1 = str(sent[i-1][0])
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            'is_date':is_a_date(word),
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = str(sent[i+1][0])
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            'is_date':is_a_date(word),
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

def sentence2input(sentence):
    temp = [list(row)for row in custom_pos_tagger(custom_tokenizer(sentence))]
    new_temp =[]
    for row in temp:
        if row[0] in symbols:
            new_row = [row[0],'SYM']
            new_temp.append(new_row)
        elif row[0].lower() in noun_list:
            new_row = [row[0],'NNP']
            new_temp.append(new_row)
        elif is_a_date(row[0]):
            new_row = [row[0],'CD']
            new_temp.append(new_row)
        else:
            new_temp.append(row)
    return sent2features(new_temp)