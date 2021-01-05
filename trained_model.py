

import re  # For preprocessing
import pandas as pd  # For data handling
from collections import defaultdict  # For word frequency
import spacy  # For preprocessing


df = pd.read_csv('result.csv')


df.head()


df.isnull().sum()



nlp = spacy.load('en_core_web_sm')  # loading the spacy for english lamguage


# Linguistic Preprocessing
def clean(sen):
    text = [token.lemma_ for token in sen if
            not token.is_stop]  # This removes stop-words from our sentences and do lemmatization
    if len(text) > 2:   # Removing words less than 3 words
        return ' '.join(text)



more_clean = (re.sub("[^A-Za-z']+", ' ', str(row)).lower() for row in
              df['text'])  # To remove words with wrong characters and lowering all



# Calling function and spacy library for preprocessing
text = [clean(doc) for doc in nlp.pipe(more_clean, batch_size=5000, n_threads=-1)]



df_cleansent = pd.DataFrame({'clean': text})  # Clean sentences
df_cleansent.shape


u = df['label']
u.shape

# In[15]:


df_cleansent = df_cleansent.join(u)  # adding labels back to all sentences

# In[16]:


df_cleansent = df_cleansent.dropna().drop_duplicates()  # dropping NULL and duplicates sentences from column
df_cleansent.shape

# In[17]:


sentences = [row.split() for row in df_cleansent['clean']]




word_freq = defaultdict(int)  # Defining the dictionary
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
len(word_freq)



sorted(word_freq, key=word_freq.get, reverse=True)[:10]

# In[20]:


import multiprocessing

from gensim.models import Word2Vec

# In[21]:


cores = multiprocessing.cpu_count()

# In[29]:


# Defining the word2vec model
# Taking window size as 3
w2v_model = Word2Vec(min_count=4, window=3, size=300, sample=6e-5, alpha=0.03, min_alpha=0.0007, negative=20,

              workers=cores - 1)
w2v_model.save("modelname.model")
#w2v_model=Word2Vec.load("modelname.model")

# Building the vocabulary using word2vec
w2v_model.build_vocab(sentences, progress_per=10000)


# Train word2vec on our data to get vectors reprenstation for each words
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=25, report_delay=1)



w2v_model.init_sims(replace=True)  # precomputing L2-norms of word weight vectors


w2v_model.wv.most_similar(positive=["law"])  # most similar to law word


w2v_model.wv.most_similar(positive=["justice"])  ##most similar to justice word



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
vocab_size, embedding_size = w2v_weights.shape  # vocavulary size and dimension of each words
print("Vocabulary Size: {} - Embedding Dim: {}".format(vocab_size, embedding_size))



sentences



arr = np.array(sentences)
ls = arr[0]
ls



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
label = {r: w for r, w in zip(new_sent.categories, range(len(new_sent.categories)))}
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

# In[72]:


label

# In[65]:


import matplotlib.pyplot as plt

ValPercent = 0.15  # Percentage of Validation set

sample = setx.shape[0]
ValN = int(ValPercent * sample)  # No.of validation data
TrainN= sample - ValN  # No. of train data

#Randomize distribution
random_i = random.sample(range(sample), sample)  # to randomize the distribution from each class
TrainX = setx[random_i[:TrainN]]
TrainY = sety[random_i[:TrainN]]
ValX = setx[random_i[TrainN:TrainN+ ValN]]
ValY = sety[random_i[TrainN:TrainN+ ValN]]

print("Train Shapes - X: {} - Y: {}".format(TrainX.shape, TrainY.shape))
print("Val Shapes - X: {} - Y: {}".format(ValX.shape, ValY.shape))

#Plotting the distribution for training set
categories, ccount = np.unique(TrainY, return_counts=True)
plt.figure(figsize=(16, 8))
plt.title("Training Set - Category Distribution")
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

# In[66]:


# Keras Embedding layer with Word2Vec weights initialization
# model = Sequential()
# model.add(
#     Embedding(input_dim=vocab_size, output_dim=embedding_size, weights=[w2v_weights], input_length=MaxSeqLength,
#               mask_zero=True,
#               trainable=False))
#
# # model.add(Bidirectional(LSTM(100)))
# model.add(LSTM(32, return_sequences=True, input_shape=(8, 16)))
# # returns a sequence of vectors of dimension 32
# # model.add(Dropout(0.2))
# # returns a sequence of vectors of dimension 32
# model.add(LSTM(32))
# model.add(Dropout(0.2))
# model.add(Dense(n_categories, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # bilstm = model.fit(TrainX, TrainY, epochs=27, batch_size=16,validation_data=(ValX, ValY), verbose=1)


# In[76]:


model = Sequential()
#Keras layer with word2vec embedding
model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_size,
                    weights=[w2v_weights],
                    input_length=141,
                    mask_zero=True,
                    trainable=False))
model.add(Bidirectional(LSTM(32, return_sequences=True, input_shape=(8, 16))))
# returns a sequence of vectors of dimension 32
model.add(Dropout(0.2))
# returns a sequence of vectors of dimension 32
model.add(Bidirectional(LSTM(50)))
model.add(Dropout(0.2))
model.add(Dense(n_categories, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model = model.fit(TrainX, TrainY, epochs=30, batch_size=16,validation_data=(ValX, ValY), verbose=1)
# In[43]:


model.save_weights("model10.h5")
#model.load_weights("learn_embb.h5")

#print("saved")

# In[45]:


#bilstm.bilstm['accuracy']

# In[46]:

#
# plt.figure(figsize=(12, 12))
# plt.plot(bilstm.bilstm['loss'])
# plt.plot(bilstm.bilstm['val_loss'])
# plt.title('Loss')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#
# plt.figure(figsize=(12, 12))
# plt.plot(bilstm.bilstm['accuracy'])
# plt.plot(bilstm.bilstm['val_accuracy'])
# plt.title('Accuracy')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

# In[49]:

#Load the best weights


# In[54]:

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

    # print(value)
    prob[x] = ind

    x = x + 1
    # k=k+1
# print("size")
prob = np.array(prob)

# In[51]:


labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

# In[53]:

#Printing ConfusionMatrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ValY, prob)
cm

# In[55]:

#Printing Classifcation Report
from sklearn.metrics import classification_report
matrix = classification_report(ValY, prob, labels=[0, 1, 2, 3, 4, 5, 6])
print('Classification report : \n', matrix)





