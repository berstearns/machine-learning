import sys
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Embedding
from keras.preprocessing import sequence
import keras.models

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
    env["models_dir"] = os.path.join(env["data_dir"],"models")
    env["vocab_dir"] = os.path.join(env["data_dir"],"vocabularies")
    return env# }}}

def load_models(env,nTokens_inDoc,nDocsToUse,minFreq_ofWords):# {{{
    vocab_filename = "vocab_"+str(minFreq_ofWords)+"_"+str(nDocsToUse)
    vocab_abspath = os.path.join(env["vocab_dir"],vocab_filename)

    filehandler = open(vocab_abspath,"rb")
    vocab = pickle.load(filehandler)

    model_filename = "LSTM_"+str(nTokens_inDoc)+"_"+str(len(vocab))
    model_abspath = os.path.join(env["models_dir"],model_filename)
    return keras.models.load_model(model_abspath),vocab# }}}

def load_corpus(env,trainOrTest,nDocsToUse):# {{{
    corpus = []
    corpus_folder = os.path.join(env["preproc_dir"],trainOrTest)
    for root,dirs,files in os.walk(corpus_folder):
        for idx,filename in enumerate(files):
            if idx > nDocsToUse:
                break
            if filename != ".DS_Store":
                abspath = os.path.join(root,filename)
                filehandler = open(abspath,"rb")
                preproc_doc = pickle.load(filehandler)
                corpus.append(preproc_doc)
    return corpus# }}}

def createVocab_fromCorpus(env,corpus,nDocsToUse,minFreq_ofWords):# {{{

    # min_df defines a minimum number of times each token must have been used
    cv_model = CountVectorizer(min_df = minFreq_ofWords)
    cv_model.fit(corpus)
    vocabulary = cv_model.vocabulary_

    vocab_filename = "vocab_"+str(minFreq_ofWords)+"_"+str(nDocsToUse)
    vocab_abspath = os.path.join(env["vocab_dir"],vocab_filename)
    filehandler = open(vocab_abspath,"wb")
    pickle.dump(vocabulary,filehandler)

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
    model.fit(corpus_train, corpus_train, epochs=2, batch_size=32)
    return model# }}}

def createLSTM_model(env,corpus,nTokens_inDoc,nDocsToUse,minFreq_ofWords):# {{{
    vocab = createVocab_fromCorpus(env,corpus,nDocsToUse,minFreq_ofWords)
    corpus_vocabIdx = mapTokens_toVocabIdx(corpus,vocab,nTokens_inDoc)
    corpus_vocabIdx = sequence.pad_sequences(corpus_vocabIdx,\
            maxlen=nTokens_inDoc, padding="post")
    corpus_train = createSequenceRepresentation(corpus_vocabIdx,\
            vocab,nTokens_inDoc)
    model = trainLSTM(corpus_train,vocab,nTokens_inDoc)
    model_filename = "LSTM_"+str(nTokens_inDoc)+"_"+str(len(vocab))
    model_abspath = os.path.join(env["models_dir"],model_filename)
    model.save(model_abspath)# }}}

def useLSTM_model(corpus,vocab,model,nTokens_inDoc):# {{{
    corpus_vocabIdx = mapTokens_toVocabIdx(corpus,vocab,nTokens_inDoc)
    corpus_vocabIdx = sequence.pad_sequences(corpus_vocabIdx,\
            maxlen=nTokens_inDoc, padding="post")
    corpus_train = createSequenceRepresentation(corpus_vocabIdx,\
            vocab,nTokens_inDoc)
    numericalLSTM_representation = []
    for idx,docMatrix in enumerate(model.predict(corpus_train)):
        numericalLSTM_representation.append(sum(docMatrix)/len(docMatrix))
    return numericalLSTM_representation# }}}

if __name__ == "__main__":# {{{
    env = configEnv()
    trainOrTest = sys.argv[1]
    params = {param_str.split("=")[0]:int(param_str.split("=")[1]) for param_str in sys.argv[2:]}
    
    nDocsToUse = params["nDocsToUse"]
    minFreq_ofWords = params["minFreq_ofWords"]
    nTokens_inDoc = params["nTokens_inDoc"]

    if trainOrTest == "train":
        corpus = load_corpus(env,trainOrTest,nDocsToUse)
        createLSTM_model(env,corpus,nTokens_inDoc,nDocsToUse,minFreq_ofWords)
    elif trainOrTest == "test":
        nDocsToTrain = 999999
        corpus = load_corpus(env,trainOrTest,nDocsToTrain)
        LSTM,vocab = load_models(env,nTokens_inDoc,nDocsToUse,minFreq_ofWords)
        X = useLSTM_model(corpus,vocab,LSTM,nTokens_inDoc)
        X = pd.DataFrame(X)

        categories = ['alt.atheism', 'soc.religion.christian']
        class_names = ['atheism', 'christian']
        y = pd.Series(fetch_20newsgroups(subset="test", categories=categories).target)
        X["y"] = y
        mlData_filename = "dataset.csv"
        mlData_abspath = os.path.join(env["preproc_dir"],"ml",mlData_filename)
        X.to_csv(mlData_abspath,index=False,header=False)
    else:
        raise Exception("Didn't find option")# }}}
