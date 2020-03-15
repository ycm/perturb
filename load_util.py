import unicodedata, string, pickle
import numpy as np
import tensorflow.keras as K
from nltk.translate.bleu_score import corpus_bleu

def load_data(filepath):
    '''
    Sentence pairs from Tatoeba project have the following format:
    English + TAB + The Other Language + TAB + Attribution
    '''
    with open(filepath, 'rb') as f:
        return [line.decode('utf-8').strip().split('\t')[:-1] for line in f]

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
        
def encode_and_pad(X):
    maxlen = max(len(s.split()) for s in X)
    T = K.preprocessing.text.Tokenizer()
    T.fit_on_texts(X)
    X_enc = T.texts_to_sequences(X)
    X_enc_pad = K.preprocessing.sequence.pad_sequences(X_enc, maxlen=maxlen, padding='post')
    return X_enc_pad, T

def vocab_len(X):
    return len({num for seq in X for num in seq})

def onehot_3d(X, vocab_len):
    onehot = np.array([K.utils.to_categorical(seq, vocab_len) for seq in X])
    return onehot

def train_split(X, Y, train_size=.9):
    assert len(X) == len(Y)
    cutoff = int(len(X) * train_size)
    return X[:cutoff], X[cutoff:], Y[:cutoff], Y[cutoff:]

def bleu(true, pred):
    weights = [
        (1,     0,   0,   0),
        (1/2, 1/2,   0,   0),
        (1/3, 1/3, 1/3,   0),
        (1/4, 1/4, 1/4, 1/4)
    ]
    return [corpus_bleu(true, pred, w) for w in weights]

def evaluate(model, X, Y, X_tokenizer, Y_tokenizer):
    SRC_idx2word = {v:k for k, v in X_tokenizer.word_index.items()}
    TGT_idx2word = {v:k for k, v in Y_tokenizer.word_index.items()}
    predictions = model.predict(X)
    y = []
    y_pred = []
    for i, pred in enumerate(predictions):
        src_sent = [SRC_idx2word[idx] for idx in X[i] if idx in SRC_idx2word]
        tgt_sent = [TGT_idx2word[idx] for idx in Y[i] if idx in TGT_idx2word]
        tgt_pred = [np.argmax(val) for val in pred]
        tgt_pred = [TGT_idx2word[idx] for idx in tgt_pred if idx in TGT_idx2word]
        y.append([tgt_sent])
        y_pred.append(tgt_pred)
        if i < 20:
            print(' '.join(tgt_sent), ' --> ', ' '.join(tgt_pred))
    bleu_scores = bleu(y, y_pred)
    for idx, bleu_score in enumerate(bleu_scores):
        print('{}-gram BLEU: {:.4f}'.format(idx + 1, bleu_score))

def evaluate_script(fp, model, X, Y, X_tokenizer, Y_tokenizer, header='No header'):
    SRC_idx2word = {v:k for k, v in X_tokenizer.word_index.items()}
    TGT_idx2word = {v:k for k, v in Y_tokenizer.word_index.items()}
    predictions = model.predict(X)
    y = []
    y_pred = []
    for i, pred in enumerate(predictions):
        src_sent = [SRC_idx2word[idx] for idx in X[i] if idx in SRC_idx2word]
        tgt_sent = [TGT_idx2word[idx] for idx in Y[i] if idx in TGT_idx2word]
        tgt_pred = [np.argmax(val) for val in pred]
        tgt_pred = [TGT_idx2word[idx] for idx in tgt_pred if idx in TGT_idx2word]
        y.append([tgt_sent])
        y_pred.append(tgt_pred)
    bleu_scores = bleu(y, y_pred)
    print(header)
    print(header, file=fp)
    for idx, bleu_score in enumerate(bleu_scores):
        print('{}-gram BLEU:\t{:.6f}'.format(idx + 1, bleu_score))
        print('{}-gram BLEU:\t{:.6f}'.format(idx + 1, bleu_score), file=fp)