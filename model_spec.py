import tensorflow.keras as K

def build_model(
    input_vocab_len,
    input_max_len,
    output_vocab_len,
    output_max_len,
    embed_dim,
    lstm1_units,
    lstm2_units):
    
    model = K.models.Sequential()
    model.add(K.layers.Embedding(
        input_dim=input_vocab_len,
        output_dim=embed_dim,
        mask_zero=True,
        input_length=input_max_len))
    model.add(K.layers.LSTM(units=lstm1_units))
    model.add(K.layers.RepeatVector(n=output_max_len))
    model.add(K.layers.LSTM(units=lstm2_units, return_sequences=True))
    model.add(K.layers.TimeDistributed(K.layers.Dense(output_vocab_len, activation='softmax')))
    
    return model