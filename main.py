from definitions import *
from utils.data_manager import *
from spacy.lang.zh import Chinese
from spacy.lang.fi import Finnish
from nltk.corpus import stopwords

stop_en = set(stopwords.words('english'))

#character segmentation default
nlp_zh = Chinese()
nlp_fi = Finnish()

# Jieba
cfg = {"segmenter": "jieba"}
nlp = Chinese.from_config({"nlp": {"tokenizer": cfg}})
# PKUSeg with "mixed" model provided by pkuseg
cfg = {"segmenter": "pkuseg"}
nlp = Chinese.from_config({"nlp": {"tokenizer": cfg}})
#nlp.tokenizer.initialize(pkuseg_model="mixed")

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
        
        updates = clean(df["reference"], lemmatize=preprocess_config['lemmatize'], stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'], stop=preprocess_config['stop'])
        update_df(df, updates, "reference")

        updates = clean(df["translation"], lemmatize=preprocess_config['lemmatize'], stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'], stop=preprocess_config['stop'])
        update_df(df, updates, "translation")

        number_token(df)
        df = tokenize(df)

        final_df.append(run_models(df))
        correlations.append(evaluate_models(df))  

    ### preprocess chinese and finnish dataframe

    language_list_from_english = language_list[-1:-2]

    for df in language_list_from_english:

    #en_zh['reference_token'] = [jieba.cut(x, cut_all=False) for x in df['reference']]
    #en_zh['translation_token'] = [jieba.cut(x, cut_all=False) for x in df['translation']] 
             
        updates = clean(df["reference_token"].tolist(), lemmatize=preprocess_config['lemmatize'], stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'], stop=preprocess_config['stop'])
        update_df(df, updates, "reference_token")

        updates = clean(df["translation_token"].tolist(), lemmatize=preprocess_config['lemmatize'], stemmer=preprocess_config['stemmer'], stop_words=preprocess_config['stop_words'], stop=preprocess_config['stop'])
        update_df(df, updates, "translation_token")

        number_token(df)
        df = tokenize(df)

        final_df.append(run_models(df))
        correlations.append(evaluate_models(df)) 






