'''
    * parse raw text into python reprensetation
    * segment text:
        1. paragraphs as units of document structure
        1. in sentences as units of discourse
    * tokenization:
        1. generate tokens as units(atoms) of semantics

'''
from nltk import sent_tokenize
from nltk import wordpunct_tokenize
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
import os
import sys
import pickle

def configEnv():# {{{
    env = {}
    env["base_dir"] = os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) )
    env["data_dir"] = os.path.join(env["base_dir"], "data")
    env["preproc_dir"] = os.path.join(env["data_dir"],"preprocessed_data")
    return env# }}}
def tokenize_paragraph(paragraph):# {{{
        PS = PorterStemmer()
        LM = WordNetLemmatizer()
        preproc_paragraph = []
        for sentence in paragraph:
            tokens = wordpunct_tokenize(sentence)
            cleaned_tokens = [ PS.stem(LM.lemmatize(token)) for token in tokens ]
            pos_tokens = nltk.pos_tag(cleaned_tokens)
            tokens = [tupl_[0] for tupl_ in pos_tokens]
            preproc_paragraph.append(" ".join(tokens))
        return preproc_paragraph# }}}

def tokenize_doc(doc):# {{{
    tokenizer = RegexpTokenizer(r'\w+')
    preproc_doc = []
    paragraphs = doc.replace("\n","$$").split("$$")
    onlyWords_paragraphs = [" ".join(tokenizer.tokenize(paragraph)) for paragraph in paragraphs]

    for paragraph in onlyWords_paragraphs:
        segmentedParagraph = sent_tokenize(paragraph)
        tknz_paragraph = tokenize_paragraph(segmentedParagraph)
        tknz_paragraph = " ".join(tknz_paragraph)
        preproc_doc.append(tknz_paragraph)
    return preproc_doc# }}}
if __name__ == "__main__":
    trainOrTest = sys.argv[1]
    env = configEnv()
    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_set = fetch_20newsgroups(subset=trainOrTest, categories=categories)
    class_names = ['atheism', 'christian']

    preprocessed_docs = []
    for doc in newsgroups_set.data:
        preproc_doc = tokenize_doc(doc)
        preproc_doc = " ".join(preproc_doc)
        preprocessed_docs.append(preproc_doc)

    preprocessedDocs_abspath = os.path.join( env["preproc_dir"]\
            ,trainOrTest,"preproc_docs")
    for idx,pre_doc in enumerate(preprocessed_docs):
        doc_abspath = preprocessedDocs_abspath+str(idx)+".pickle"
        filehandler = open(doc_abspath,"wb")
        pickle.dump(pre_doc,filehandler)
