# Import some libraries
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
seed = 0
np.random.seed(seed)
import seaborn as sns
sns.set(style = 'whitegrid')
import nest_asyncio
nest_asyncio.apply()
import re
import string
import json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from keras.preprocessing.text import tokenizer_from_json
from keras.utils import pad_sequences
from keras.models import load_model

#load dataset
tweets_data = pd.read_csv('dataset_clean_sentiment.csv')
# tweets = tweets_data[['id', 'username', 'created_at', 'tweet', 'text_clean', 'text_preprocessed', 'sentiment_score', 'sentiment']]

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) 
    text = re.sub(r'#[A-Za-z0-9]+', '', text) 
    text = re.sub(r'RT[\s]', '', text) 
    text = re.sub(r"http\S+", '', text)
    text = re.sub(r'[0-9]+', '', text)

    text = text.replace('\n', ' ')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip(' ')
    return text

def casefoldingText(text): 
    text = text.lower() 
    return text

def tokenizingText(text):
    text = word_tokenize(text) 
    return text

def filteringText(text): 
    listStopwords = set(stopwords.words('indonesian'))
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered 
    return text

def stemmingText(text): 
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    return text

def toSentence(list_words):
    sentence = ' '.join(word for word in list_words)
    return sentence

X = tweets_data['text_preprocessed'].apply(toSentence)
max_features = 5000
with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)

X = tokenizer.texts_to_sequences(X.values)
X = pad_sequences(X)
print(X.shape)

model = load_model("model_modifikasi.h5")

input_text = 'pengen kuliah online ampe lulus'
text = [input_text]
new_text = []
for text in text:
    preprocessed_text = toSentence(stemmingText(filteringText(tokenizingText
                        (casefoldingText(cleaningText(text))))))
    new_text.append(preprocessed_text)
print(new_text)
text_seq = tokenizer.texts_to_sequences(new_text)
text_seq = pad_sequences(text_seq, maxlen = 47)
text_pred = model.predict(text_seq)
text_pred = np.argmax(text_pred)
print(X.shape)
print(text_pred)
if text_pred == 0:
    text_result = 'Negatif'
elif text_pred == 1:
    text_result = 'Neutral'
else:
    text_result ='Positive'
print('Hasil prediksi sentiment = ', text_result)