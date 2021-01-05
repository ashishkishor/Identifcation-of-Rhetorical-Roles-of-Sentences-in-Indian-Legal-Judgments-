import re  # For preprocessing
import pandas as pd  # For data handling
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import spacy  # For preprocessing


df = pd.read_csv('result.csv')
df.head()
df.isnull().sum()
nlp = spacy.load('en_core_web_sm')  # Spacy to load the english language


# Linguistic Preprocessing
def clean(sen):
    text = [token.lemma_ for token in sen if
            not token.is_stop]  # This removes stop-words from our sentences and do lemmatization
    if len(text) > 2:  # Removing words less than 3 words
        return ' '.join(text)



more_clean = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in
              df['text'])  # To remove words with wrong characters and lowering all

# Calling function and spacy library for preprocessing
text = [clean(doc) for doc in nlp.pipe(more_clean, batch_size=5000, n_threads=-1)]
df_cleansent = pd.DataFrame({'clean': text})  # Clean sentences
df_cleansent.shape
u=df['label']
u.shape
df_cleansent = df_cleansent.join(u)  # Adding labels at the back of sentences
df_cleansent = df_cleansent.dropna().drop_duplicates()  #This is to drop NULL and blank columns
df_cleansent.shape

sentences = [row.split() for row in df_cleansent['clean']]


word_freq = defaultdict(int)  # Defining the dictionary
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq)

sorted(word_freq, key=word_freq.get, reverse=True)[:10]

import multiprocessing
cores = multiprocessing.cpu_count()

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

w2v_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

#Importing all necessary libraries for our model
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
import sys
import numpy as np
import string
import logging
import random

#Initializing weights
w2v_weights = w2v_model.wv.vectors
VocabSize, EmbeddingSize = w2v_weights.shape #Vocavulary size and dimension of each words
print("The Vocabulary Size is : {} -  Embedding Dim: {}".format(VocabSize, EmbeddingSize))

#Function to extract word from word2vec
def word2token(word):
    try:
        return w2v_model.wv.vocab[word].index
    except KeyError:
        return 0


#Class to convert the sentences to match with the index in vocabulory which corresponds to vector of that word
MaxSeqLength = 200

class Sent2vec:
    def __init__(self, data, seq_length):
        self.data = data

        self.categories = data.label.unique()
        self.seq_length = seq_length

    def __iter__(self):
        for txt, cat in zip(self.data.iloc[:, 0], self.data.iloc[:, 1]):
            words = np.array([word2token(w) for w in txt.split(' ')[:self.seq_length] if w != ''])
            yield (words, cat)

new_sent = Sent2vec(df_cleansent, MaxSeqLength)


#Generate the labels
label = {r: m for r, m in zip(new_sent.categories, range(len(new_sent.categories)))}
""" {'Facts': 0,
 'Ratio of the decision': 1,
 'Ruling by Lower Court': 2,
 'Argument': 3,
 'Ruling by Present Court': 4,
 'Precedent': 5,
 'Statute': 6}
 """

setx = []
sety = []
for w, c in new_sent:
    setx.append(w)
    sety.append(label[c])

# Padding to equalize the vectors
setx = pad_sequences(setx, maxlen=MaxSeqLength, padding='pre', value=0)
sety = np.array(sety)
print(setx.shape)
print(sety.shape)

import  matplotlib.pyplot as plt

valPer = 0.15 # Percentage of Validation set
Samples = setx.shape[0]
ValN = int(valPer * Samples) # No.of validation data
TrainN = Samples - ValN  # No. of train data

#Randomize distribution
random_i = random.sample(range(Samples), Samples)
TrainX = setx[random_i[:TrainN]]
TrainY = sety[random_i[:TrainN]]
ValX = setx[random_i[TrainN:TrainN+ValN]]
ValY = sety[random_i[TrainN:TrainN+ValN]]

print(TrainX)
print(TrainY)
print(ValX)
print(ValY)
print(random_i)

print("Training sample shapes - X: {} - Y: {}".format(TrainX.shape, TrainY.shape))
print("Validation sample shapes - X: {} - Y: {}".format(ValX.shape, ValY.shape))

#Plotting the distribution for training set
categories, ccount = np.unique(TrainY, return_counts=True)
plt.figure(figsize=(16, 8))
plt.title("Training Set - ""Category Distribution")
plt.xticks(range(len(categories)), label.keys())
plt.bar(categories, ccount, align='center')
plt.show()
#Plotting the distribution for validation set
categories, ccount = np.unique(ValY, return_counts=True)
plt.figure(figsize=(16, 8))
plt.title("Validation Set - Category Distribution")
plt.xticks(range(len(categories)), label.keys())
plt.bar(categories, ccount, align='center')
plt.show()

n_categories = len(categories)

#Keras layer with word2vec embedding
model = Sequential()
model.add(Embedding(input_dim=VocabSize,
                    output_dim=EmbeddingSize,
                    weights=[w2v_weights],
                    input_length=MaxSeqLength,
                    mask_zero=True,
                    trainable=False))

#model.add(LSTM(6, return_sequences=True, input_shape=(4, 8)) )
# returns a sequence of vectors of dimension 32
#model.add(Dropout(0.2))
# returns a sequence of vectors of dimension 32
# model.add(LSTM(6))

model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.4))
model.add(Dense(n_categories, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(TrainX, TrainY, epochs=6, batch_size=8,
                   validation_data=(ValX, ValY), verbose=1)
#Load the best weights
#model.load_weights('model13.h5')
#
# plt.figure(figsize=(12, 12))
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Loss')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#
# plt.figure(figsize=(12, 12))
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Accuracy')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#model.save_weights("model13.h5")

#Predicting on validation set
result = model.predict(ValX)
i = 0
from sklearn import metrics

# print(metrics.classification_report(ValY,result))
# result =model.predict(TrainX)
# converting the result into 1 hot vector

prob = [None] * 1158
x = 0
y = 0

print("loop")
# k=0
index = 0
for i in result:
    maxx = -1
    y = 0
    # if(k>10):
    # break
    for j in i:

        # print(*i)
        if maxx < j:
            maxx = j
            value = j
            ind = y
        y = y + 1
    prob[x] = ind

    x = x + 1
prob = np.array(prob)

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

#Printing ConfusionMatrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ValY, prob)
cm

#Printing Classifcation Report
from sklearn.metrics import classification_report
matrix = classification_report(ValY, prob, labels=[0, 1, 2, 3, 4, 5, 6])
print('Classification report : \n', matrix)


