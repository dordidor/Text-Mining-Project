from definitions import *
from utils.data_manager import *
from spacy.lang.zh import Chinese
from spacy.lang.fi import Finnish
from nltk.corpus import stopwords

stop_en = set(stopwords.words('english'))

def new_func(df):
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    df = df.dropna()
    return df

if __name__ == '__main__':

    language_list = load_dataset()
    preprocess_config = {
        'lemmatize': False,
        'stemmer': False,
        'stop_words': False,
        'stop': stop_en
        # lowercase
        # remove punctuation
        }

    # replace numbers with a token ##

    final_df=[]
    correlations = []

    # preprocess all dataframes

    language_list_to_en = language_list[0:-2]

    for df in language_list_to_en:
        
        df = new_func(df)

        updates = clean(df["reference"], lower = True, lemmatize=preprocess_config['lemmatize'], stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'], stop=preprocess_config['stop'])
        update_df(df, updates, "reference")

        updates = clean(df["translation"], lower = True, lemmatize=preprocess_config['lemmatize'], stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'], stop=preprocess_config['stop'])
        update_df(df, updates, "translation")

        number_token(df)
        df = tokenize(df)

        final_df.append(run_models(df))
        correlations.append(evaluate_models(df))  

    ### preprocess chinese and finnish dataframe

    en_zh = pd.read_csv("corpus/en-zh/scores.csv")
    en_fi = pd.read_csv("corpus/en-fi/scores.csv")

    language_list_from_english = language_list[-1:-3]

    #for df in language_list_from_english:

        #df = new_func(df)
             
        #updates = clean(df["reference"].tolist(), lower = True, lemmatize=preprocess_config['lemmatize'], stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'], stop=preprocess_config['stop'])
        #update_df(df, updates, "reference")

        #updates = clean(df["translation"].tolist(), lower = True, lemmatize=preprocess_config['lemmatize'], stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'], stop=preprocess_config['stop'])
        #update_df(df, updates, "translation")

        #number_token(df)
        #df = tokenize(df)

        #en_zh['reference_token'] = [jieba.cut(x, cut_all=False) for x in df['reference']]
        #en_zh['translation_token'] = [jieba.cut(x, cut_all=False) for x in df['translation']] 

        #final_df.append(run_models(df))
        #correlations.append(evaluate_models(df)) 






