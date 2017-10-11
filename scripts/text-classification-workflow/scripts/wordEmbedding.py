import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Embedding
from keras.preprocessing import sequence

import h5py
import pickle
import os
import re
from nltk.tokenize import RegexpTokenizer

def configEnv():# {{{
    env = {}
    env["base_dir"] = os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) )
    env["data_dir"] = os.path.join(env["base_dir"], "data")
    env["preproc_dir"] = os.path.join(env["data_dir"],"preprocessed_data")
    return env# }}}
def load_corpus(env):# {{{
    corpus = []
    for root,dirs,files in os.walk(env["preproc_dir"]):
        for filename in files:
            if filename != ".DS_Store":
                abspath = os.path.join(root,filename)
                filehandler = open(abspath,"rb")
                preproc_doc = pickle.load(filehandler)
                corpus.append(preproc_doc)
    return corpus# }}}

def createVocab_fromCorpus(corpus):# {{{

    # min_df defines a minimum number of times each token must have been used
    minFreq_ofWords = 5
    cv_model = CountVectorizer(min_df = minFreq_ofWords)
    cv_model.fit(corpus)

    vocabulary = cv_model.vocabulary_
    return vocabulary# }}}

def mapTokens_toVocabIdx(corpus,vocabulary,nTokens_inDoc):# {{{
    corpusAs_vocabIdx = []
    tokenizer = RegexpTokenizer(r'\w+')
    for doc in corpus:
        count = 0
        doc_indexedTokens = []
        doc = tokenizer.tokenize(doc)
        for token in doc:
            if count < nTokens_inDoc:
                token = token.lower()
                try:tokenIdx_inVocab = vocabulary[token]
                except:pass
                doc_indexedTokens.append(tokenIdx_inVocab)
                count += 1
        corpusAs_vocabIdx.append(doc_indexedTokens)
    return corpusAs_vocabIdx# }}}

def createSequenceRepresentation(corpus_vocabIdx,vocabulary,nTokens_inDoc):# {{{
    corpus_train = []
    for docIdx in range(len(corpus_vocabIdx)):
        doc = corpus_vocabIdx[docIdx]
        doc_matrixRepresentation = np.zeros((nTokens_inDoc, len(vocabulary)))
        for tokenIdx in range(len(doc)):
            vocabIdx = doc[tokenIdx]
            doc_matrixRepresentation[tokenIdx][vocabIdx] = 1
        corpus_train.append(doc_matrixRepresentation)
    return np.array(corpus_train)# }}}
def trainLSTM(corpus_train,vocabulary,nTokens_inDoc):# {{{
    model = Sequential()
    model.add(LSTM(len(vocabulary), input_dim = len(vocabulary), input_length=nTokens_inDoc, return_sequences=True))
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(corpus_train, corpus_train, epochs=10, batch_size=32)
    return model# }}}
if __name__ == "__main__":
    env = configEnv()
    corpus = load_corpus(env)
    vocab = createVocab_fromCorpus(corpus)
    nTokens_inDoc = 20
    corpus_vocabIdx = mapTokens_toVocabIdx(corpus,vocab,nTokens_inDoc)
    corpus_vocabIdx = sequence.pad_sequences(corpus_vocabIdx,\
            maxlen=nTokens_inDoc, padding="post")
    corpus_train = createSequenceRepresentation(corpus_vocabIdx,\
            vocab,nTokens_inDoc)
    model = trainLSTM(corpus_train,vocab,nTokens_inDoc)
    model_abspath = os.path.join(env["data_dir"],"mymodel.h5")
    model.save(model_abspath)

