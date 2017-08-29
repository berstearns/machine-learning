from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Embedding
from keras.preprocessing import sequence

import h5py
import os
import re

def build_tokenizer():
    token_pattern = r"(?u)\b\w\w+\b"
    token_pattern = re.compile(token_pattern)
    return lambda doc: token_pattern.findall(doc)


addr = "./aclImdb/train/unsup"

corpus = []
print('Reading dataset...')
for filename in os.listdir(addr)[:500]:
    f = open(addr+'/'+filename)
    corpus.append(f.read())
    f.close()

# take the first nPhrasesTaken. Ignore the rest
nPhrasesTaken = 200
corpus = corpus[:nPhrasesTaken]

print('Creating vocabulary...')

# min_df defines a minimum number of times each word must have been used
minFreq_ofWords = 10
cv_model = CountVectorizer(min_df = minFreq_ofWords)
cv_model.fit(corpus)

vocabulary = cv_model.vocabulary_

''' each phrase is a list of idx's '''
phrasesList = []
# takes in consideration only the first nWords_inPhrase
nWords_inPhrase = 20
for phrase in corpus:
	'''for each word in the phrase, append the idx of the word in the vocabulary 
	   in the order of appearance in the phrase '''
	wordsIdxList = []
	wordsList = build_tokenizer()(phrase)
	count = 0
	for word in wordsList:
		if count < nWords_inPhrase:
			try:
			    word = word.lower()
			    wordIdx_inVocab = vocabulary[word]
			    wordsIdxList.append(wordIdx_inVocab)
			except:
				pass
			finally:
				count += 1
	phrasesList.append(wordsIdxList)


''' makes all phrases lists of the same size , still list of idx's ''' 
# padding='post' pad zeros to the right instead of padding to the left
phrasesList = sequence.pad_sequences(phrasesList, maxlen=nWords_inPhrase, padding='post')

phrases_train = []
for phrase_idx in range(len(phrasesList)):
	phrase = phrasesList[  phrase_idx  ]

	phrase_MatrixRepresentation = np.zeros((nWords_inPhrase, len(vocabulary)))

	for word_idx in range(len(phrase)):
	    vocabIdx = phrasesList[phrase_idx][word_idx]
	    phrase_MatrixRepresentation   [word_idx]  [vocabIdx] = 1
	phrases_train.append(  phrase_MatrixRepresentation  )


phrases_train = np.array(phrases_train)
print('Training...')
print(len(vocabulary))

model = Sequential()

model.add(LSTM(len(vocabulary), input_dim = len(vocabulary), input_length=nWords_inPhrase, return_sequences=True))
model.compile(optimizer='rmsprop', loss='mse')
model.fit(phrases_train, phrases_train, epochs=10, batch_size=32)
model.save("mymodel.h5")