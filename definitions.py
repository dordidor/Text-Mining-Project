from mods.models import meteor, nist, baseline_bleu, bleu_rouge, rouge_1, charf, sacre_bleu
from tqdm import tqdm_notebook as tqdm
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from bs4 import BeautifulSoup
import re
from matplotlib import pyplot
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from metrics import *
import string

#stop_en = set(stopwords.words('english'))
#stop_fi = set(stopwords.words('finnish'))
# stop_zh = chinese library needed 
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
snowball_stemmer = SnowballStemmer('english')

def number_token(df):

    def transform_number(text):
        """
        Function that receives a string of text and returns the string with 
        the cost formats within it substituted by the token #COST
        """
        tokenized_text = re.sub('(\d+|\d+.\d+)(| )','##',text)
            
        return tokenized_text

    df["reference"] = [transform_number(x) for x in df["reference"]]
    df["translation"] = [transform_number(x) for x in df["translation"]]

def tokenize(df):
    df['reference_token'] = [[x.split()] for x in df['reference']]
    df['translation_token'] = [x.split() for x in df['translation']]
    return df

def clean(text_list, lemmatize=False, stemmer=False, punctuation = False, stop_words=False, stop = ["a"]):
    """
    Function that a receives a list of strings and preprocesses it.
    
    :param text_list: List of strings.
    :param lemmatize: Tag to apply lemmatization if True.
    :param stemmer: Tag to apply the stemmer if True.
    """
    updates = []
    for j in tqdm(range(len(text_list))):
        
        text = text_list[j]

        #LOWERCASE TEXT
        text = text.lower()
        
        #REMOVE NUMERICAL DATA AND PUNCTUATION
        if punctuation:
            text = re.sub("[^a-zA-Z]", ' ', text)
        
        #REMOVE TAGS (HTML)
        text = BeautifulSoup(text, features='lxml').get_text()
        
        #REMOVE STOP WORDS - not needed 
        #if stop_words:
            #text = " ".join([word for word in text.split() if word not in stop])
        
        #LEMMATIZATION
        if lemmatize:
            text = " ".join(lemma.lemmatize(word) for word in text.split())
        
        #STEMMER
        if stemmer:
            text = " ".join(snowball_stemmer.stem(word) for word in text.split())
        
        updates.append(text)
        
    return updates

def update_df(dataframe, list_updated, column):
    return dataframe.update(pd.DataFrame({column: list_updated}))


def total_word_freq(text_list):
    """
    Function that receives a list of strings and returns the frequency of each word
    in the set of all strings.
    """
    words_in_df = ' '.join(text_list).split()
    # Count all words 
    freq = pd.Series(words_in_df).value_counts()
    return freq

# Fetch wordcount for each column
def word_count(df):
    word_count_ref  = df['reference'].apply(lambda x: len(str(x).split(" ")))
    word_count_tra  = df['translation'].apply(lambda x: len(str(x).split(" ")))
    df['word_count_ref'] = word_count_ref
    df['word_count_tra'] = word_count_tra

def get_top_n_grams(corpus, top_k, n):
    """
    Function that receives a list of documents (corpus) and extracts
        the top k most frequent n-grams for that corpus.
        
    :param corpus: list of texts
    :param top_k: int with the number of n-grams that we want to extract
    :param n: n gram type to be considered 
             (if n=1 extracts unigrams, if n=2 extracts bigrams, ...)
             
    :return: Returns a sorted dataframe in which the first column 
        contains the extracted ngrams and the second column contains
        the respective counts
    """
    vec = CountVectorizer(ngram_range=(n, n), max_features=2000).fit(corpus)
    
    bag_of_words = vec.transform(corpus)
    
    sum_words = bag_of_words.sum(axis=0) 
    
    words_freq = []
    for word, idx in vec.vocabulary_.items():
        words_freq.append((word, sum_words[0, idx]))
        
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    top_df = pd.DataFrame(words_freq[:top_k])
    top_df.columns = ["Ngram", "Freq"]
    return top_df

def run_models(df):
    # get word count for each of reference and translation
    word_count(df)

    # apply baseline bleu model
    baseline_bleu(df)

    # apply sacre bleu
    sacre_bleu(df)

    # apply NIST model
    #nist(df)

    # apply the rouge model
    rouge_1(df)

    #apply the bleu-rouge f1 
    bleu_rouge(df)

    #apply meteor model
    meteor(df)

    #apply charF
    charf(df)

    return df

def evaluate_models(df): # TODO for laser
    model_list = ['bleu','sacre_bleu','rouge','bleu_rouge','meteor','charf']  
    correl_df = []
    #set indices
    for model in model_list:
        reg = RegressionReport()
        correl_df[model] = reg.compute(df[model], df['z-score'])

    return correl_df

def plot_correl(df,column):
    pyplot.scatter(df['z-score'], df[column])
    pyplot.show()












