import csv
from math import ceil
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import ModelCheckpoint

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

    time_step = []
    sunspots = []

    with open('sunspots.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            sunspots.append(float(row[2]))
            time_step.append(int(row[0]))

    series = np.array(sunspots)

    # DO NOT CHANGE THIS CODE
    # This is the normalization function
    min = np.min(series)
    max = np.max(series)
    series -= min
    series /= max
    time = np.array(time_step)

    # The data should be split into training and validation sets at time step 3000
    # DO NOT CHANGE THIS CODE
    split_time = 3000

    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    # DO NOT CHANGE THIS CODE
    window_size = 30
    batch_size = 32
    shuffle_buffer_size = 1000

    train_set = windowed_dataset(x_train, window_size=window_size, batch_size=batch_size,
                                 shuffle_buffer=shuffle_buffer_size)

    validation_set = windowed_dataset(x_valid, window_size=window_size, batch_size=batch_size,
                                      shuffle_buffer=shuffle_buffer_size)

    model = Sequential([
        Conv1D(filters=64, kernel_size=5, padding="causal", activation="relu", input_shape=[None, 1]),
        LSTM(64, return_sequences=True),
        LSTM(64, return_sequences=True),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1)
    ])

    model.summary()

    # PLEASE NOTE IF YOU SEE THIS TEXT WHILE TRAINING -- IT IS SAFE TO IGNORE
    # BaseCollectiveExecutor::StartAbort Out of range: End of sequence
    #     [[{{node IteratorGetNext}}]]
    #
    optimizer = SGD(lr=1e-5, momentum=0.9)

    checkpoint_path = 'sunspots_normal_checkpoint.ckpt'

    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1
                                 )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                 patience = 2,
                                                 factor = 0.8,
                                                 verbose=1)

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3,
                                                     verbose=1)
    model.compile(loss=Huber(),
                  optimizer=optimizer,
                  metrics=["mae"],
                  )

    model.fit(train_set, validation_data=(validation_set), epochs=200, callbacks=[checkpoint, reduce_lr, earlystopping])

    model.load_weights(checkpoint_path)

    # YOUR CODE HERE TO COMPILE AND TRAIN THE MODEL

    return model, series, x_valid

if __name__ == '__main__':
    model = solution_model()[0]
    model.save("sunspots_normal_model.h5")

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