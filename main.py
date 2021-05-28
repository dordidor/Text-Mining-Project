from definitions import *
from utils.data_manager import *
import jieba

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

    language_list_to_en = language_list[:-2]

    list_of_names = ['cs-en', 'de-en', 'ru-en', 'zh-en', 'en-fi', 'en-zh']

    # To English section
    for name, df in enumerate(language_list_to_en):

        print("Cleaning " + list_of_names[name])

        updates = clean(df["reference"], lower=True, lemmatize=preprocess_config['lemmatize'], stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'], stop=preprocess_config['stop'])
        update_df(df, updates, "reference")

        updates = clean(df["translation"], lower=True, lemmatize=preprocess_config['lemmatize'], stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'], stop=preprocess_config['stop'])
        update_df(df, updates, "translation")

        df = remove_empty(df)

        number_token(df)
        df = tokenize(df)

        print("Running models for " + list_of_names[name])
        final_df.append(run_models(df, list_of_names[name]))
        correlations.append(evaluate_models(df))


    # To Chinese section
    preprocess_config["stop"] = stop_zh

    print("Cleaning " + list_of_names[-1])

    language_list[-1]['reference_token'] = language_list[-1]['reference'].astype('unicode').fillna(u'NA')
    language_list[-1]['reference_token'] = language_list[-1]['reference_token'].apply(lambda x: " ".join([r[0] for r in jieba.tokenize(x)]))

    language_list[-1]['translation_token'] = language_list[-1]['translation'].astype('unicode').fillna(u'NA')
    language_list[-1]['translation_token'] = language_list[-1]['translation_token'].apply(lambda x: " ".join([r[0] for r in jieba.tokenize(x)]))

    print("Running models for " + list_of_names[-1])
    final_df.append(run_models(language_list[-1], list_of_names[-1]))
    correlations.append(evaluate_models(language_list[-1]))

    # To Finnish section
    preprocess_config["stop"] = stop_fi
    print("Cleaning " + list_of_names[-2])
    updates = clean(language_list[-2]["reference"].tolist(), lemmatize=preprocess_config['lemmatize'],
                    stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'],
                    stop=preprocess_config['stop'])
    update_df(language_list[-2], updates, "reference")

    updates = clean(language_list[-2]["translation"].tolist(), lemmatize=preprocess_config['lemmatize'],
                    stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'],
                    stop=preprocess_config['stop'])
    update_df(language_list[-2], updates, "translation")

    number_token(language_list[-2])
    df = tokenize(language_list[-2])

    print("Running models for " + list_of_names[-2])
    final_df.append(run_models(language_list[-2], list_of_names[-2]))
    correlations.append(evaluate_models(language_list[-2]))

    for i in range(len(correlations)):
        print("Correlations for", list_of_names[i])
        correlations[i].to_csv("results/" + list_of_names[i] + "_config1.csv")
        print(correlations[i])
