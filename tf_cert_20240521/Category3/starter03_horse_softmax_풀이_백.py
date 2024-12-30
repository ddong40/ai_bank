# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Computer Vision with CNNs
#
# This task requires you to create a classifier for horses or humans using
# the provided dataset.
#
# Please make sure your final layer has 2 neurons, activated by softmax
# as shown. Do not change the provided output layer, or tests may fail.
#
# IMPORTANT: Please note that the test uses images that are 300x300 with
# 3 bytes color depth so be sure to design your input layer to accept
# these, or the tests will fail.
#

import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

dataset_name = 'horses_or_humans'
train_dataset, info = tfds.load(name=dataset_name, split='train', with_info=True)
valid_dataset = tfds.load(name=dataset_name, split='test')

def preprocess(features):
    # YOUR CODE HERE
    # image, label = tf.image.convert_image_dtype(features['image'], tf.float32), features['label']
    image = tf.cast(features['image'], tf.float32) / 255.0
    image = tf.image.resize(image, (300, 300))
    label = tf.one_hot(features['label'], depth=2)

    return image, label


def solution_model():
    train_data = train_dataset.map(preprocess).batch(32).repeat()
    valid_data = valid_dataset.map(preprocess).batch(256).repeat()

    train_image_size = info.splits['train'].num_examples
    steps_per_epoch = (train_image_size) // 32 + 1

    valid_image_size = info.splits['test'].num_examples
    validation_steps = (valid_image_size) // 256

    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(300, 300, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        # YOUR CODE HERE, BUT MAKE SURE YOUR LAST LAYER HAS 2 NEURONS ACTIVATED BY SOFTMAX
        Dense(2, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(0.0001)

    model.summary()

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint_path = 'my_checkpoint.ckpt'

    checkpoint = ModelCheckpoint(checkpoint_path,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             verbose=1
                             )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  patience=10,
                                  factor=0.8,
                                  verbose=1)

    earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)

    model.fit(train_data,
              steps_per_epoch=steps_per_epoch,
              validation_data=(valid_data),
              validation_steps=validation_steps,
              epochs=1,
              callbacks=[checkpoint, reduce_lr, earlystopping],
              verbose=2
              )

    model.load_weights(checkpoint_path)

    return model

# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("mymodel.h5")
