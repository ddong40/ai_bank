# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# NLP QUESTION
#
# For this task you will build a classifier for the sarcasm dataset
# The classifier should have a final layer with 1 neuron activated by sigmoid as shown
# It will be tested against a number of sentences that the network hasn't previously seen
# And you will be scored on whether sarcasm was correctly detected in those sentences

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/sarcasm.json'
    urllib.request.urlretrieve(url, 'sarcasm.json')

    # DO NOT CHANGE THIS CODE OR THE TESTS MAY NOT WORK
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []

    # YOUR CODE HERE
    with open('sarcasm.json', 'r') as f:
        datastore = json.load(f)

    for item in datastore:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    train_sentences = sentences[:training_size]
    train_labels = labels[:training_size]

    validation_sentences = sentences[training_size:]
    validation_labels = labels[training_size:]

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    # word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded = pad_sequences(validation_sequences, padding=padding_type, maxlen=max_length)

    train_labels = np.array(train_labels)
    validation_labels = np.array(validation_labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(24, activation='relu'),
        # YOUR CODE HERE. KEEP THIS OUTPUT LAYER INTACT OR TESTS MAY FAIL
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    checkpoint_path = 'sarcasm_checkpoint.ckpt'

    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1,)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  patience=2,
                                  factor=0.5,
                                  verbose=1,)

    earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_padded, train_labels,
              validation_data=(validation_padded, validation_labels),
              callbacks=[checkpoint, reduce_lr, earlystopping],
              epochs=6,
              verbose=1)

    model.load_weights(checkpoint_path)

    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("sarcasm_model.h5")
