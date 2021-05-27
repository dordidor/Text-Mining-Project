import pandas as pd
import numpy as np


def load_dataset():
    # dataframes provided with the corresponding language translations

    # assign dataset names
    list_of_names = ['cs-en', 'de-en', 'ru-en', 'zh-en', 'en-fi', 'en-zh']

    # create empty list
    dataframes_list = []
  
    # append datasets into teh list
    for i in range(len(list_of_names)):

        temp_df = pd.read_csv("corpus/"+ list_of_names[i]+"/scores.csv")

        dataframes_list.append(temp_df)

    return dataframes_list


def load_embeddings():
    # assign dataset names
    list_of_names = ['cs-en', 'de-en', 'ru-en', 'zh-en', 'en-fi', 'en-zh']

    # create empty list
    embedding_ref_list = []
    embedding_tra_list = []

    # append datasets into teh list
    for i in range(len(list_of_names)):
        temp_ref_df = pd.read_csv("corpus/" + list_of_names[i] + "/laser.reference_embeds.npy")
        temp_tra_df = pd.read_csv("corpus/" + list_of_names[i] + "/laser.translation_embeds.npy")

        embedding_ref_list.append(temp_ref_df)
        embedding_tra_list.append(temp_tra_df)

    return embedding_ref_list, embedding_tra_list
