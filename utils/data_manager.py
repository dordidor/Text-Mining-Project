import os
import pandas as pd
import numpy as np
import urllib.request

from random import sample, seed
from shutil import rmtree
from PIL import Image
from glob import glob


def load_dataset():
    # dataframes provided with the corresponding language translations

    # assign dataset names
    list_of_names = ['cs-en', 'de-en', 'en-fi', 'en-zh', 'ru-en', 'zh-en', 'en-fi', 'en-zh']

    # create empty list
    dataframes_list = []
  
    # append datasets into teh list
    for i in range(len(list_of_names)):

        temp_df = pd.read_csv("corpus/"+ list_of_names[i]+"/scores.csv")

        dataframes_list.append(temp_df)

    return dataframes_list
