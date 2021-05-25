from definitions import *
from utils.data_manager import *
from spacy.lang.zh import Chinese
from spacy.lang.fi import Finnish
from nltk.corpus import stopwords

stop_en = set(stopwords.words('english'))

def remove_empty(df):
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    df = df.dropna()
    return df

if __name__ == '__main__':

    language_list = load_dataset()
    preprocess_config = {
        'lemmatize': False,
        'stemmer': False,
        'punctuation': True,
        'stop_words': False,
        'stop': stop_en
        # lowercase
        # remove punctuation
        }

    # replace numbers with a token ##

    final_df=[]
    correlations = []

    # preprocess all dataframes

    language_list_to_en = language_list[2:3]

    list_of_names = ['cs-en', 'de-en', 'ru-en', 'zh-en', 'en-fi', 'en-zh']

    for name, df in enumerate(language_list_to_en):

        df_size = df.shape[0]
        print ("Cleaning "+ list_of_names[name])

        updates = clean(df["reference"], lower = True, lemmatize=preprocess_config['lemmatize'], stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'], stop=preprocess_config['stop'])
        update_df(df, updates, "reference")

        updates = clean(df["translation"], lower = True, lemmatize=preprocess_config['lemmatize'], stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'], stop=preprocess_config['stop'])
        update_df(df, updates, "translation")

        df = remove_empty(df)
        print(df.shape[0]/df_size)

        number_token(df)
        df = tokenize(df)

        print("Running models for "+ list_of_names[name])
        final_df.append(run_models(df))
        correlations.append(evaluate_models(df))  

    





