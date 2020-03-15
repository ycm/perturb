import unicodedata, string, pickle
import numpy as np

def load_data(filepath):
    '''
    Sentence pairs from Tatoeba project have the following format:
    English + TAB + The Other Language + TAB + Attribution
    '''
    with open(filepath) as f:
        return [line.strip().split('\t')[:-1] for line in f]

def preprocess(dataset):
    ''' Processes a dataset loaded using load_data. '''
    punct = str.maketrans('', '', string.punctuation)
    return np.array([[preprocess_string(s, punct) for s in pair] for pair in dataset])

def preprocess_string(_s, punct):
    '''
    1. Unicode normalize
    2. Encode as ASCII
    2. Minuscule
    3. Remove punctuation
    '''
    s = unicodedata.normalize('NFKC', _s).encode('ascii', 'ignore').decode('utf-8')
    s = s.lower()
    s = s.translate(punct)
    return s

def dump_data(data, filepath):
    ''' Pickle data file. '''
    with open(filepath, 'wb') as f:
        pickle.dump(data, filepath)
