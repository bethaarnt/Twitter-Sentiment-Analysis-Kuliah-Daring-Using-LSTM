# Import some libraries

import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
seed = 0
np.random.seed(seed)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid')
import nest_asyncio
nest_asyncio.apply()
import datetime as dt
import re
import string
import io, json
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Embedding, Dense, Dropout, LSTM
from keras.optimizers import Adam, RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

tweets_data = pd.read_csv('dataset.csv')
tweets = tweets_data[['id', 'username', 'created_at', 'tweet']]

def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
    text = re.sub(r'RT[\s]', '', text) # remove RT
    text = re.sub(r"http\S+", '', text) # remove link
    text = re.sub(r'[0-9]+', '', text) # remove numbers

    text = text.replace('\n', ' ') # replace new line into space
    text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
    text = text.strip(' ') # remove characters space from both left and right text
    return text

tweets['text_clean'] = tweets['tweet'].apply(cleaningText)

def casefoldingText(text): # Converting all the characters in a text into lower case
    text = text.lower() 
    return text

tweets['text_clean'] = tweets['text_clean'].apply(casefoldingText)

def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
    text = word_tokenize(text) 
    return text

tweets['text_preprocessed'] = tweets['text_clean'].apply(tokenizingText)

def filteringText(text): # Remove stopwors in a text
    listStopwords = set(stopwords.words('indonesian'))
    filtered = []
    for txt in text:
        if txt not in listStopwords:
            filtered.append(txt)
    text = filtered 
    return text

tweets['text_preprocessed'] = tweets['text_preprocessed'].apply(filteringText)

def stemmingText(text): # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = [stemmer.stem(word) for word in text]
    return text

def toSentence(list_words): # Convert list of words into sentence
    sentence = ' '.join(word for word in list_words)
    return sentence

tweets.to_csv(r'dataset_clean.csv', index = False, header = True, index_label=None)
tweets

# Because preprocessing tweets data takes a lot time, so I load tweets data which has been preprocessed before
tweets = pd.read_csv('dataset_clean.csv')

for i, text in enumerate(tweets['text_preprocessed']):
    tweets['text_preprocessed'][i] = tweets['text_preprocessed'][i].replace("'", "")\
                                            .replace(',','').replace(']','').replace('[','')
    list_words=[]
    for word in tweets['text_preprocessed'][i].split():
        list_words.append(word)
        
    tweets['text_preprocessed'][i] = list_words   
# Determine sentiment of tweets using indonesia sentiment lexicon (source : https://github.com/fajri91/InSet)

# Loads lexicon positive and negative data
lexicon_positive = dict()
import csv
with open('lexicon_positive.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])

lexicon_negative = dict()
import csv
with open('lexicon_negative.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_negative[row[0]] = int(row[1])
        
# Function to determine sentiment sentiment of tweets        
def sentiment_analysis_lexicon_indonesia(text):
    #for word in text:
    score = 0
    for word in text:
        if (word in lexicon_positive):
            score = score + lexicon_positive[word]
    for word in text:
        if (word in lexicon_negative):
            score = score + lexicon_negative[word]
    sentiment=''
    if (score > 0):
        sentiment = 'positive'
    elif (score < 0):
        sentiment = 'negative'
    else:
        sentiment = 'neutral'
    return score, sentiment

# Results from determine sentiment sentiment of tweets

results = tweets['text_preprocessed'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
tweets['sentiment_score'] = results[0]
tweets['sentiment'] = results[1]
print(tweets['sentiment'].value_counts())

# Export to csv file
tweets.to_csv(r'dataset_clean_sentiment.csv', index = False, header = True,index_label=None)

## Analysis and Visualization
fig, ax = plt.subplots(figsize = (6, 6))
sizes = [count for count in tweets['sentiment'].value_counts()]
labels = list(tweets['sentiment'].value_counts().index)
explode = (0.01, 0.01, 0.01)
ax.pie(x = sizes, labels = labels, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
ax.set_title('Sentiment Labels on Tweets Data \n (total = 25K Tweet data)', fontsize = 16, pad = 20)
plt.show()

# Visualize word cloud

list_words=''
for tweet in tweets['text_preprocessed']:
    for word in tweet:
        list_words += ' '+(word)
        
wordcloud = WordCloud(width = 600, height = 400, background_color = 'black', min_font_size = 10).generate(list_words)
fig, ax = plt.subplots(figsize = (8, 6))
ax.set_title('Word Cloud of Tweets Data', fontsize = 18)
ax.grid(False)
ax.imshow((wordcloud))
fig.tight_layout(pad=0)
ax.axis('off')
plt.show()

# Function to group all positive/negative words
def words_with_sentiment(text):
    positive_words=[]
    negative_words=[]
    for word in text:
        score_pos = 0
        score_neg = 0
        if (word in lexicon_positive):
            score_pos = lexicon_positive[word]
        if (word in lexicon_negative):
            score_neg = lexicon_negative[word]
        
        if (score_pos + score_neg > 0):
            positive_words.append(word)
        elif (score_pos + score_neg < 0):
            negative_words.append(word)
            
    return positive_words, negative_words

# Visualize positive and negative word cloud

sentiment_words = tweets['text_preprocessed'].apply(words_with_sentiment)
sentiment_words = list(zip(*sentiment_words))
positive_words = sentiment_words[0]
negative_words = sentiment_words[1]

fig, ax = plt.subplots(1, 2,figsize = (12, 10))
list_words_postive=''
for row_word in positive_words:
    for word in row_word:
        list_words_postive += ' '+(word)
wordcloud_positive = WordCloud(width = 800, height = 600, background_color = 'black', colormap = 'Greens'
                            , min_font_size = 10).generate(list_words_postive)
ax[0].set_title('Word Cloud of Positive Words on Tweets Data \n (based on Indonesia Sentiment Lexicon)', fontsize = 14)
ax[0].grid(False)
ax[0].imshow((wordcloud_positive))
fig.tight_layout(pad=0)
ax[0].axis('off')

list_words_negative=''
for row_word in negative_words:
    for word in row_word:
        list_words_negative += ' '+(word)
wordcloud_negative = WordCloud(width = 800, height = 600, background_color = 'black', colormap = 'Reds'
                            , min_font_size = 10).generate(list_words_negative)
ax[1].set_title('Word Cloud of Negative Words on Tweets Data \n (based on Indonesia Sentiment Lexicon)', fontsize = 14)
ax[1].grid(False)
ax[1].imshow((wordcloud_negative))
fig.tight_layout(pad=0)
ax[1].axis('off')

plt.show()

# Make text preprocessed (tokenized) to untokenized with toSentence Function

X = tweets['text_preprocessed'].apply(toSentence) 
max_features = 5000

# Tokenize text with specific maximum number of words to keep, based on word frequency
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(X.values)

tokenizer_json = tokenizer.to_json()
with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))
X = tokenizer.texts_to_sequences(X.values)
X = pad_sequences(X)
X.shape

# Encode target data into numerical values
sentiment_encode = {'negative' : 0, 'neutral' : 1, 'positive' : 2}
y = tweets['sentiment'].map(sentiment_encode).values

# Split the data (with composition data train 80%, data test 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# model lstm
model = Sequential()
model.add(Embedding(input_dim = max_features, output_dim = 64, input_length = X_train.shape[1]))
# model.add(embedding_layer)
model.add(LSTM(units = 16, activation = 'tanh'))
model.add(Dropout(0.1))
model.add(Dense(units = 3, activation = 'softmax'))
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = RMSprop(learning_rate = 0.001), metrics = ['accuracy'])
print(model.summary())

model_prediction = model.fit(X_train, y_train, epochs=8, batch_size=128, validation_split=0.1)

# Visualization model accuracy (train and val accuracy)

fig, ax = plt.subplots(figsize = (10, 4))
ax.plot(model_prediction.history['accuracy'], label = 'train accuracy')
ax.plot(model_prediction.history['val_accuracy'], label = 'val accuracy')
ax.set_title('Model Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend(loc = 'upper left')
plt.show()

# Predict sentiment on data test by using model has been created, and then visualize a confusion matrix
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy on Test Data:', accuracy)
confusion_matrix(y_test, y_pred)
print('F1-Score:', f1_score(y_test, y_pred, average=None))

fig, ax = plt.subplots(figsize = (8,6))
sns.heatmap(confusion_matrix(y_true = y_test, y_pred = y_pred), fmt = 'g', annot = True)
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Prediction', fontsize = 14)
ax.set_xticklabels(['negative (0)', 'neutral (1)', 'positive (2)'])
ax.set_ylabel('Actual', fontsize = 14)
ax.set_yticklabels(['negative (0)', 'neutral (1)', 'positive (2)'])
plt.show()
print(classification_report(y_test, y_pred))

#Save model
model.save('model_modifikasi.h5')

## Results from prediction sentiment on data test
text_clean = tweets['text_clean']
text_train, text_test = train_test_split(text_clean, test_size = 0.2, random_state = 0)

result_test = pd.DataFrame({'text': text_test, 
                            'aktual': y_test,
                            'prediction': y_pred})
sentiment_decode = {0 : 'Negative', 1 : 'Neutral', 2 : 'Positive'}
result_test['aktual'] = result_test['aktual'].map(sentiment_decode)
result_test['prediction'] = result_test['prediction'].map(sentiment_decode)
pd.set_option('max_colwidth', 300)
result_test

#Predict with Other Data
# Initializing and preprocessing new text data
otherData = pd.DataFrame()
otherData['text'] = ['enaknya kuliah online, ga perlu capek" ke kampus dan bisa santai dirumah',
                    'Tapi serius deh semakin kesini ngerasa kuliah online makin ga efektif, ga paham materi blasss, kopong, berasa yauda kek ga kuliah', 
                    'tumbenan td kuliah online dosennya minta join zoom, trs temen ngcht nyuruh masuk krn yg lain udh join, tinggal gua doang yg blm, Okedeh....   cepet cepet,cuma pakek daster + kemeja (ditimpa gt daster gua) + pasmina🙂  Jadilaaaa',
                    'Ya allah hebat kali aing ya dulu gapyear sekarang kuliah online. Berasa nganggur 2 tahun ngab',
                    'Kalo kuliah online berasa sia sia njirr, bayar ukt elit nikmati fasilitas sulit'
                    ]
otherData['text_clean'] = otherData['text'].apply(cleaningText)
otherData['text_clean'] = otherData['text_clean'].apply(casefoldingText)
otherData.drop(['text'], axis = 1, inplace = True)

otherData['text_preprocessed'] = otherData['text_clean'].apply(tokenizingText)
otherData['text_preprocessed'] = otherData['text_preprocessed'].apply(filteringText)
otherData['text_preprocessed'] = otherData['text_preprocessed'].apply(stemmingText)
otherData

# Preprocessing text data
# Make text preprocessed (tokenized) to untokenized with toSentence Function
#Load model
model = load_model('model_modifikasi.h5')
X_otherData = otherData['text_preprocessed'].apply(toSentence)
X_otherData = tokenizer.texts_to_sequences(X_otherData.values)
X_otherData = pad_sequences(X_otherData, maxlen = X.shape[1])

# Results from prediction sentiment on text data
y_pred_otherData = model.predict(X_otherData)
y_pred_otherData = np.argmax(y_pred_otherData, axis=1)
otherData['Result Prediction'] = y_pred_otherData

sentiment_decode = {0 : 'Negative', 1 : 'Neutral', 2 : 'Positive'}
otherData['Result Prediction'] = otherData['Result Prediction'].map(sentiment_decode)
otherData