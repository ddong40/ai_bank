# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Basic Datasets Question
#
# Create a classifier for the MNIST dataset
# Note that the test will expect it to classify 10 classes and that the
# input shape should be the native size of the MNIST dataset which is
# 28x28 monochrome. Do not resize the data. Your input layer should accept
# (28,28) as the input shape only. If you amend this, the tests will fail.
#

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

def solution_model():
    mnist = tf.keras.datasets.mnist

    # YOUR CODE HERE

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    print(x_train.shape)
    print(x_test.shape)

    x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))

    x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

    model = Sequential([Flatten(input_shape=(28, 28, 1)),
                        Dense(512, activation='relu'),
                        Dense(256, activation='relu'),
                        Dense(10, activation='softmax')
                        ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=512, validation_data=(x_test, y_test), epochs=5)

    model.summary()

    predict = model.predict([x_test[0:10, 0:28, 0:28, 0:1]])

    print(y_test[:10])

    print(predict)

    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")