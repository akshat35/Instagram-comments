import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk import word_tokenize,stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob, Word, Blobber
import pandas as pd
import re

Comment = pd.read_csv("dataset.csv")
Rev_comm=Comment["text"]

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

clean_comments=preprocess(Comment)
comm_words=tokenization_w(clean_comments)
word_len=len(comm_words)
lem=lemmatization(comm_words,word_len)
print(lem)

new_comment = []
for i in lem:
    new_comment.append(" ".join(i))
print(new_comment)

data = pd.DataFrame({'Comments':new_comment})
print(data['Comments'])

data['word_count'] = data['Comments'].apply(lambda x: len(str(x).split(" ")))
data[['Comments','word_count']].head()

data['char_count'] = data['Comments'].str.len() ## this also includes spaces
data[['Comments','char_count']].head()

def avg_word(sentence):
  words = sentence.split()
  print(words)
  print(len(words))
  print(sum(len(word) for word in words))
  if(len(words)==0):
    ave_w=(sum(len(word) for word in words)/(len(words)+1))
  else:
    ave_w=(sum(len(word) for word in words)/len(words))
  return ave_w

data['avg_word'] = data['Comments'].apply(lambda x: avg_word(x))
data[['Comments','avg_word']].head()

stop = stopwords.words('english')

data['stopwords'] = data['Comments'].apply(lambda x: len([x for x in x.split() if x in stop]))
data[['Comments','stopwords']].head()

data['hastags'] = data['Comments'].apply(lambda x: len([x for x in x.split() if x.startswith('#')]))
data[['Comments','hastags']].head()

data['numerics'] = data['Comments'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
data[['Comments','numerics']].head()

data['upper'] = data['Comments'].apply(lambda x: len([x for x in x.split() if x.isupper()]))
data[['Comments','upper']].head()

pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
                print(ppo, tup)
    except:
        pass
    return cnt

data['noun_count'] = data['Comments'].apply(lambda x: check_pos_tag(x, 'noun'))
data['verb_count'] = data['Comments'].apply(lambda x: check_pos_tag(x, 'verb'))
data['adj_count'] = data['Comments'].apply(lambda x: check_pos_tag(x, 'adj'))
data['adv_count'] = data['Comments'].apply(lambda x: check_pos_tag(x, 'adv'))
data['pron_count'] = data['Comments'].apply(lambda x: check_pos_tag(x, 'pron'))
data[['Comments','noun_count','verb_count','adj_count', 'adv_count', 'pron_count' ]].head()

data.head()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

cv=CountVectorizer()
A_vec = cv.fit_transform(new_comment)
print(A_vec.toarray())

tv=TfidfVectorizer()
t_vec = tv.fit_transform(new_comment)
print(t_vec.toarray())

feature_names = tv.get_feature_names()

dense = t_vec.todense()
denselist = dense.tolist()
daf = pd.DataFrame(denselist, columns=feature_names)
print(feature_names)

df_c =pd.concat([daf,data], axis=1)
df_c.head()