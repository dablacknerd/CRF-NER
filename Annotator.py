import os
from crf_feature_functions import *
from nltk import word_tokenize
import pickle

class Annotator(object):
    def __init__(self,sentence):
        diseases_model_file = os.path.join(os.getcwd(),'models','disease','disease-ner.sav')
        diseases_model = pickle.load(open(diseases_model_file,'rb'))
        
        anatomy_model_file = model_file = os.path.join(os.getcwd(),'models','anatomy-1','anatomy1-ner.sav')
        anatomy_model = pickle.load(open(model_file,'rb'))
        
        self.__sentence = sentence
        self.__tokens = word_tokenize(sentence)
        self.__disease_labels = diseases_model.best_estimator_.predict_single(sentence2input(sentence))
        self.__anatomical_labels = anatomy_model.predict_single(sentence2input(sentence))
    
    def anatomical_entities(self):
        entities =[]
        temp1 = []
        temp2 = []
        temp3 =[]
        for token,label in zip(self.__tokens,self.__anatomical_labels):
            temp1.append([token,label])
        
        for item in temp1:
            #print(item)
            if item[1].startswith('B') or item[1].startswith('I'):
                temp2.append(item[0])
            else:
                if len(temp2) == 0:
                    continue
                else:
                    temp3.append(' '.join(temp2))
                    temp2 =[]
            
        if len(temp3) > 0:
            for anatomy_entity in temp3:
                entities.append(anatomy_entity.replace(" 's","'s"))
            return entities
        else:
            return entities
    
    def diseases(self):
        diseases =[]
        temp1 = []
        temp2 = []
        temp3 =[]
        for token,label in zip(self.__tokens,self.__disease_labels):
            temp1.append([token,label])
        
        for item in temp1:
            #print(item)
            if item[1].startswith('B') or item[1].startswith('I'):
                temp2.append(item[0])
            else:
                if len(temp2) == 0:
                    continue
                else:
                    temp3.append(' '.join(temp2))
                    temp2 =[]
            
        if len(temp3) > 0:
            for disease in temp3:
                diseases.append(disease.replace(" 's","'s"))
            return diseases
        else:
            return diseases
    
    def get_annotations(self):
        entities =[]
        
        if len(self.diseases()) > 0:
            for disease_entity in self.diseases():
                start = self.__sentence.find(disease_entity)
                end = start + len(disease_entity)
                entities.append([disease_entity,start,end,"DISEASE-ENTITY"])
                
        if len(self.anatomical_entities()) > 0:
            for anatomical_entity in self.anatomical_entities():
                start = self.__sentence.find(anatomical_entity)
                end = start + len(anatomical_entity)
                entities.append([anatomical_entity,start,end,"ANATOMICAL-ENTITY"])
        
        return entities