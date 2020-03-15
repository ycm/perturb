from load_util import *
from model_spec import build_model

import argparse
import numpy as np
import tensorflow.keras as K
from nltk.translate.bleu_score import corpus_bleu

p = argparse.ArgumentParser(description='Run NMT model.')
p.add_argument('parallel_corpus', type=str, help='Parallel corpus, e.g. Tatoeba project.')
p.add_argument('output_results_file', type=str, help='Output file name to dump BLEU scores.')
p.add_argument('output_model_file', type=str, help='Output file name to dump trained model.')

p.add_argument('--reverse_translate', type=int, default=0, help='Translate second language to first language (default: 0, translates first language into second.)')
p.add_argument('--n_epochs', type=int, default=30, help='Number of epochs (default: 30)')
p.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
p.add_argument('--model_verbose', type=int, default=1, help='Keras model verbosity (default: 1)')
p.add_argument('--embed_dim', type=int, default=300, help='Size of embeddings (default: 300)')
p.add_argument('--lstm1_size', type=int, default=256, help='Size of first LSTM layer (default: 256)')
p.add_argument('--lstm2_size', type=int, default=256, help='Size of second LSTM layer (default: 256)')

args = p.parse_args()

MODEL_FILEPATH = args.output_model_file
EMBED_DIM = args.embed_dim
LSTM1_SIZE = args.lstm1_size
LSTM2_SIZE = args.lstm2_size

print('Loading data...')
data = preprocess(load_data(args.parallel_corpus))
data = data[:30000]
np.random.seed(0)
np.random.shuffle(data)

print('Processing source and target data...')
if args.reverse_translate:
    SRC, TGT = data[:, 1], data[:, 0]
else:
    SRC, TGT = data[:, 0], data[:, 1]
SRC_enc_pad, SRC_tokenizer = encode_and_pad(SRC)
TGT_enc_pad, TGT_tokenizer = encode_and_pad(TGT)
SRC_vocab_len = vocab_len(SRC_enc_pad)
TGT_vocab_len = vocab_len(TGT_enc_pad)
SRC_train, SRC_test, TGT_train, TGT_test = train_split(SRC_enc_pad, TGT_enc_pad, train_size=.9)

print('One-hot encoding target data...')
TGT_train_onehot = onehot_3d(TGT_train, TGT_vocab_len)
TGT_test_onehot = onehot_3d(TGT_test, TGT_vocab_len)

print('Building model...')
model = build_model(
    input_vocab_len=SRC_vocab_len,
    input_max_len=SRC_train.shape[1],
    output_vocab_len=TGT_vocab_len,
    output_max_len=TGT_train_onehot.shape[1],
    embed_dim=EMBED_DIM,
    lstm1_units=LSTM1_SIZE,
    lstm2_units=LSTM2_SIZE
)
model.compile(optimizer='adam', loss='categorical_crossentropy')
print(model.summary())

print('Running model...')
model_checkpoint = K.callbacks.ModelCheckpoint(filepath=MODEL_FILEPATH, monitor='val_loss', verbose=args.model_verbose, save_best_only=True)
model.fit(
    x=SRC_train,
    y=TGT_train_onehot,
    batch_size=args.batch_size,
    epochs=args.n_epochs,
    verbose=args.model_verbose,
    callbacks=[model_checkpoint],
    validation_data=[SRC_test, TGT_test_onehot])

print('Reloading trained model...')
trained_model = K.models.load_model(MODEL_FILEPATH)

print('Evaluating...')
out_file = open(args.output_results_file, 'w')
evaluate_script(fp=out_file, model=trained_model, X=SRC_train, Y=TGT_train, X_tokenizer=SRC_tokenizer, Y_tokenizer=TGT_tokenizer, header='Train')
evaluate_script(fp=out_file, model=trained_model, X=SRC_test, Y=TGT_test, X_tokenizer=SRC_tokenizer, Y_tokenizer=TGT_tokenizer, header='Test')
out_file.close()