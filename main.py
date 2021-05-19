from utils.data_manager import build_set_generators
from mods.models import Classifier
import json
from definitions import *
import warnings

if __name__ == '__main__':

    language_list = [cs_en, de_en, en_fi, en_zh, ru_en, zh_en, en_fi, en_zh]
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


    for df in language_list:

        df["reference"] = [number_token(x) for x in df["reference"]]
        df["translation"] = [number_token(x) for x in df["translation"]]
        #TODO
        updates = clean(df["reference"], lemmatize=preprocess_config['lemmatize'], stemmer=False, stop_words=False, stop=stop_en)
        update_df(df, updates, "reference")

        updates = clean(df["translation"], lemmatize=False, stemmer=False, stop_words=False, stop=stop_en)
        update_df(df, updates, "translation")

        final_df.append(run_models(df))
        correlations.append(evaluate_models(df))






