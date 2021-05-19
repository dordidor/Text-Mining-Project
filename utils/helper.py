import numbers
import collections
from tensorflow.keras import backend as k
import tensorflow as tf


def formatfloat(x):
    return float(x)

def pformat(dictionary, function=formatfloat):
    if isinstance(dictionary, dict):
        return type(dictionary)((key, pformat(value, function)) for key, value in dictionary.items())
    if isinstance(dictionary, collections.Container):
        return type(dictionary)(pformat(value, function) for value in dictionary)
    if isinstance(dictionary, numbers.Number):
        return function(dictionary)
    return dictionary

