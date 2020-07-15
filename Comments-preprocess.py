import spacy
import pandas as pd
import re

nlp = spacy.load('en_core_web_sm')
Comments = pd.read_csv("dataset.csv",usecols=["text"])

def preprocess(text):
    clean_data = []
    for x in (text[:]): 
        new_text = re.sub('<.*?>', '', x)   # remove HTML tags
        new_text = re.sub(r'[^\w\s]', '', new_text) # remove punc.
        new_text = re.sub(r'\d+','',new_text)# remove numbers
        new_text = new_text.lower() # lower case, .upper() for upper          
        if new_text != '':
            clean_data.append(new_text)
    return clean_data

def tokenization_w(words):
    w_new = []
    for w in (words[:]):
        w_token = nlp(w)
        if w_token != '':
            w_new.append(w_token)
    return w_new

def lemmatization(words):
    new = []
    for i in range(len(Comments)):
        lem_words = [x.nlp for x in (words[:][i])]
        new.append(lem_words)
    return new

clean_comments=preprocess(Comments)
comm_words=tokenization_w(clean_comments)
lem=lemmatization(comm_words)
print(lem)