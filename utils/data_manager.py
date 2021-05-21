#from definitions import *

import time
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import os
import pandas as pd
import numpy as np
import urllib.request

from random import sample, seed
from shutil import rmtree
from PIL import Image
from glob import glob

#path = '/corpus'   # use your path
#all_files = glob.glob(path + "/*.csv")
import os
def load_dataset(path):
    # dataframes provided with the corresponding language translations
    
    # assign dataset names
    list_of_names = ['cs-en', 'de-en', 'en-fi', 'en-zh', 'ru-en', 'zh-en', 'en-fi', 'en-zh']

    # create empty list
    dataframes_list = []
  
    # append datasets into teh list
    for i in range(len(list_of_names)):
        temp_df = pd.read_csv(os.path.join("..\\corpus\\"+list_of_names[i]+"\\scores.csv"))
        dataframes_list.append(temp_df)

    return dataframes_list
