import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
import re

Comments = pd.read_csv("dataset.csv")
Rev_comm=Comments["text"]

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
        w_token = word_tokenize(w)
        if w_token != '':
            w_new.append(w_token)
    return w_new

lemmat = WordNetLemmatizer()
def lemmatization(words,wl):
    new = []
    for i in range(wl):
        lem_words = [lemmat.lemmatize(x) for x in (words[:][i])]
        new.append(lem_words)
    return new

clean_comments=preprocess(Comments)
comm_words=tokenization_w(clean_comments)
word_len=len(comm_words)
lem=lemmatization(comm_words,word_len)
print(lem)