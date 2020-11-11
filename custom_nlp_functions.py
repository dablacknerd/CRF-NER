from nltk import word_tokenize, pos_tag
import re

def place_holder_gen(word_length):
    cover = ''
    for i in range(0,word_length):
        cover = cover + '*'
    return cover

def custom_tokenizer(text):
    cardiac_output_index_regex = re.compile(r'^\d{1,2}\.\d{0,2}\/\d{1,2}\.\d{0,2}$|^\d\/\d\.\d{1,2}$')
    blood_pressure_regex = re.compile(r'^\d{3}\/\d{2,3}$')
    tokens = []
    for token in word_tokenize(text):
        split_count = 1
        if re.match(cardiac_output_index_regex,token) or re.match(blood_pressure_regex,token):
            split_size = len(token.split('/'))
            for sub_token in token.split('/'):
                if split_count != split_size:
                    tokens.append(sub_token)
                    tokens.append("/")
                    split_count += 1
                else:
                    tokens.append(sub_token)
        else:
            tokens.append(token)
    return tokens

def custom_pos_tagger(tokens):
    symbols = ['/',']','[','*','#','$','%','^','&','(',')',
               '-','_','=','+','\\','~','`','|','{','}',
               '<','>'
              ]
    punctuations = ['.','!','?']
    temp_pos_tagged_tokens = pos_tag(tokens)
    final_pos_tagged_tokens =[]
    
    for row in temp_pos_tagged_tokens:
        if row[0] in symbols and row[1] != 'SYM':
            final_pos_tagged_tokens.append((row[0],'SYM'))
        elif row[0] in punctuations and row[1] != 'SENT':
            final_pos_tagged_tokens.append((row[0],'SENT'))
        else:
            final_pos_tagged_tokens.append(row)
    return final_pos_tagged_tokens

def my_word_tokenizer(sentence):
    temp_tokens = custom_tokenizer(sentence)
    new_tokens =[]
    for token in temp_tokens:
        if token == "``" or token == "''" or token == "''":
            new_tokens.append('"')
        else:
            new_tokens.append(token)
    return new_tokens

def word_tokenization_with_offset(sentence):
    temp_sent = sentence
    tokenized_sentence = my_word_tokenizer(sentence)
    special_tokenization =[]
    start_offset = None
    end_offset = None
    for word in tokenized_sentence:
        if word == "''" or word == "``" or word =="''" or word == "``":
            print("pulled over here")
            continue
        try:
            start_offset = temp_sent.index(word)
            end_offset = start_offset + len(word)
            place_holder = place_holder_gen(len(word))
            temp_sent = temp_sent.replace(temp_sent[start_offset:end_offset],place_holder,1)
            special_tokenization.append([word,start_offset,end_offset])
            #last_start_offset = end_offset
        except ValueError:
            old_word = word
            new_word = '"'
            print((word,new_word))
            print(sentence)
            start_offset = temp_sent.index(new_word)
            end_offset = start_offset + len(new_word)
            place_holder = place_holder_gen(len(new_word))
            temp_sent = temp_sent.replace(temp_sent[start_offset:end_offset], place_holder, 1)
            special_tokenization.append([new_word, start_offset, end_offset])
    return special_tokenization