# Question
#
# Create a classifier for the CIFAR10 dataset
# Note that the test will expect it to classify 10 classes and that the input shape should be
# the native CIFAR size which is 32x32 pixels with 3 bytes color depth

import tensorflow as tf

def solution_model():
    cifar = tf.keras.datasets.cifar10
    (x_train,y_train),(x_test,y_test) = cifar.load_data()
    # YOUR CODE HERE

    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Conv1D, Dense, Flatten, MaxPooling2D, Dropout, Activation, Conv2D,InputLayer
    import numpy as np
    from tensorflow.python.keras.layers import Conv1D, Dense, Flatten, Conv2D
    from keras.utils import to_categorical
    # from sklearn.metrics import accuracy_score
    import numpy as np
    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    from tensorflow.keras.layers import InputLayer, Conv2D, BatchNormalization, MaxPooling2D, Activation
    from tensorflow.keras.layers import Flatten, Dense, Dropout

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=(32, 32, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=30, validation_data=(x_test, y_test))

    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")

