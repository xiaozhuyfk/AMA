import logging
from keras.layers import Input, LSTM, Dense, Embedding, Merge, Bidirectional
from keras.models import Model, model_from_json, Sequential

logger = logging.getLogger(__name__)

def save_model_to_file(model, struct_file, weights_file):
    # save model structure
    model_struct = model.to_json()
    open(struct_file, 'w').write(model_struct)

    # save model weights
    model.save_weights(weights_file, overwrite=True)


def load_model(struct_file, weights_file):
    model = model_from_json(open(struct_file, 'r').read())
    model.load_weights(weights_file)
    return model

def bidirectional_lstm_with_embedding(vocab_dim, n_symbols, word_idx):
    """
    logger.info("Initializing embedding weights.")
    embedding_weights = np.zeros((n_symbols+1, vocab_dim))
    for word, index in word_idx.items():
        vector = modules.w2v.transform(word)
        if vector is not None:
            embedding_weights[index,:] = vector
        else:
            embedding_weights[index,:] = np.random.normal(0, 0.1, vocab_dim)
    """

    # assemble the model
    logger.info("Constructing Bi-directional LSTM model.")
    model = Sequential()

    """
    model.add(
        Embedding(output_dim=vocab_dim,
                  input_dim=n_symbols+1,
                  mask_zero=True,
                  weights=[embedding_weights])
    )
    """

    model.add(
        Embedding(output_dim=vocab_dim,
                  input_dim=n_symbols,
                  mask_zero=True)
    )
    model.add(
        Bidirectional(LSTM(16))
    )
    model.add(
        Dense(1)
    )
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=["accuracy"])

    return model