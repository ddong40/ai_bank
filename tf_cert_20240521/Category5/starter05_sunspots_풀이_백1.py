# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Sequence Modelling Question.
#
# For this task you will need to train a neural network
# to predict sunspot activity using the Sunspots.csv
# provided. Your neural network is expected to have an MAE
# of at least 20, with top marks going to one with an MAE
# of around 15. At the bottom is provided some testing
# code should you want to check before uploading which measures
# the MAE for you. Strongly recommend you test your model with
# this to be able to see how it performs.

import csv
import tensorflow as tf
import numpy as np
import urllib
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, LSTM, Lambda
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from math import ceil
# import matplotlib.pyplot as plt


# DO NOT CHANGE THIS CODE
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/Sunspots.csv'
    urllib.request.urlretrieve(url, 'sunspots.csv')

    # Your data should be loaded into 2 Python lists called time_step
    # and sunspots. They are decleared here.
    time_step = []
    sunspots = []

    with open('sunspots.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            sunspots.append(float(row[2]))
            time_step.append(int(row[0]))

    series = np.array(sunspots)

    # csvfile = pd.read_csv('sunspots.csv')
    # print(csvfile)
    # print(type(csvfile))

    # series = csvfile.iloc[:, 2]
    # print(series)

    # time_step = csvfile.iloc[:, 0]
    # print(time_step)

    # print(type(series))
    # print(type(time_step))

    # You should use numpy to create
    # - your series from the list of sunspots
    # - your time details from the list of time steps
    series = np.array(sunspots)
    time = np.array(time_step)

    # You should split the dataset into training and validation splits
    # At time 3000. So everything up to 3000 is training, and everything
    # after 3000 is validation. Write the code below to achieve that.
    split_time = 3000
    time_train = time[:split_time]
    x_train = series[:split_time]

    time_valid = time[split_time:]
    x_valid = series[split_time:]

    # DO NOT CHANGE THIS CODE
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000

    # You can use any random seed you want. We use 51. :)
    # tf.random.set_seed(51)
    # np.random.seed(51)
    train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size,
                                 shuffle_buffer=shuffle_buffer_size)

    validation_set = windowed_dataset(x_valid, window_size=window_size, batch_size=batch_size,
                                      shuffle_buffer=shuffle_buffer_size)

    # YOUR CODE HERE TO COMPILE AND TRAIN THE MODEL
    model = Sequential([
        Conv1D(filters=60,
               kernel_size=5,
               strides=1,
               padding="causal",
               activation="relu",
               input_shape=[None, 1]),
        LSTM(60, return_sequences=True),
        LSTM(60, return_sequences=True),
        Dense(30, activation="relu"),
        Dense(10, activation="relu"),
        # YOUR CODE HERE. DO NOT CHANGE THE FINAL TWO LAYERS FROM BELOW
        Dense(1),
        # The data is not normalized, so this lambda layer helps
        # keep the MAE in line with expectations. Do not modify.
        Lambda(lambda x: x * 400)
    ])

    model.summary()

    optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)

    checkpoint_path = 'sunspots_checkpoint.ckpt'

    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  patience=2,
                                  factor=0.8,
                                  verbose=1)

    earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)

    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=["mae"])

    model.fit(train_set,
              validation_data=(validation_set),
              epochs=200,
              callbacks=[checkpoint, reduce_lr, earlystopping])

    model.load_weights(checkpoint_path)

    return model


#
# # Note that you'll need to save your model as a .h5 like this
# # This .h5 will be uploaded to the testing infrastructure
# # and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("sunspots_model.h5")

# THIS CODE IS USED IN THE TESTER FOR FORECASTING. IF YOU WANT TO TEST YOUR MODEL
# BEFORE UPLOADING YOU CAN DO IT WITH THIS
#     def model_forecast(model, series, window_size):
#        ds = tf.data.Dataset.from_tensor_slices(series)
#        ds = ds.window(window_size, shift=1, drop_remainder=True)
#        ds = ds.flat_map(lambda w: w.batch(window_size))
#        ds = ds.batch(32).prefetch(1)
#        forecast = model.predict(ds)
#        return forecast
#
#
#     window_size = 30 # YOUR CODE HERE
#     split_time = 3000
#     series = solution_model()[1]
#     x_valid = solution_model()[2]
#
#     rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size)
#     rnn_forecast = rnn_forecast[split_time - window_size:-1, -1, 0]
#
#     result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
#
#     # WE EXPECT AN MAE OF 15 or less for the maximum score
#     score = ceil(20 - result)
#     if score > 5:
#         score = 5
#
#     print(score)
