from definitions import *
import warnings
from utils.data_manager import *
import jieba

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

    model_config = {
        'model': baseline_bleu
        }

    train_config = {
        'n_epochs': 50
                }

    # replace numbers with a token ##

    final_df=[]
    correlations = []

    language_list.remove('en-zh')

    for df in language_list:

        df = tokenize(df)
        number_token(df)

        #[jieba.cut(x, cut_all=False) for x in df['reference']]
        #[jieba.cut(x, cut_all=False) for x in df['translation']]  
        
        updates = clean(df["reference"], lemmatize=preprocess_config['lemmatize'], stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'], stop=preprocess_config['stop'])
        update_df(df, updates, "reference")

        updates = clean(df["translation"], lemmatize=preprocess_config['lemmatize'], stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'], stop=preprocess_config['stop'])
        update_df(df, updates, "translation")

        final_df.append(run_models(df))
        correlations.append(evaluate_models(df))  






