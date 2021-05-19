import nltk
import jieba
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer



def preprocessing(text):
    # split word
    token_words = word_tokenize(text)
    # filter stopwords
    # stop_words = stopwords.words('english')
    # fileter_words = [word for word in token_words if word not in stop_words]
    # stemmer
    # stemmer = PorterStemmer()
    # fileterStem_words = [stemmer.stem(word) for word in fileter_words]
    return token_words


def preprocessing_mooc(text):
    # split word
    token_words = jieba.lcut(text)
    # filter stopwords
    # stop_words = stopwords.words('english')
    # fileter_words = [word for word in token_words if word not in stop_words]
    # stemmer
    # stemmer = PorterStemmer()
    # fileterStem_words = [stemmer.stem(word) for word in fileter_words]
    return token_words


def word2num(texts, word_dict):
    nums_list = []
    for text in texts:
        num_list = []
        token_words = word_tokenize(text)
        for word in token_words:
            try:
                num_list.append(word_dict[word])
            except KeyError:
                num_list.append(0)
        nums_list.append(num_list)
    return nums_list
