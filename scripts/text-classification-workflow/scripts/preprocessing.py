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

categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
class_names = ['atheism', 'christian']

tokenizer = RegexpTokenizer(r'\w+')
PS = PorterStemmer()
LM = WordNetLemmatizer()
for doc in newsgroups_train.data:
    paragraphs = doc.replace("\n","$$").split("$$")
    paragraphs = [" ".join(tokenizer.tokenize(paragraph)) for paragraph in paragraphs]
    for paragraph in paragraphs:
        segmentedParagraph = sent_tokenize(paragraph)
        for sentence in segmentedParagraph:
            tokens = wordpunct_tokenize(sentence)
            cleaned_tokens = [ PS.stem(LM.lemmatize(token)) for token in tokens ]
            nltk.pos_tag(cleaned_tokens)
