# ======================================================================
# There are 5 questions in this test with increasing difficulty from 1-5
# Please note that the weight of the grade for the question is relative
# to its difficulty. So your Category 1 question will score much less
# than your Category 5 question.
# ======================================================================
#
# Computer Vision with CNNs
#
# For this task you will build a classifier for Rock-Paper-Scissors
# based on the rps dataset.
#
# IMPORTANT: Your final layer should be as shown, do not change the
# provided code, or the tests may fail
#
# IMPORTANT: Images will be tested as 150x150 with 3 bytes of color depth
# So ensure that your input layer is designed accordingly, or the tests
# may fail.
#
# NOTE THAT THIS IS UNLABELLED DATA.
# You can use the ImageDataGenerator to automatically label it
# and we have provided some starter code.


import urllib.request
import zipfile
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras_preprocessing.image import ImageDataGenerator

def solution_model():
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'
    urllib.request.urlretrieve(url, 'rps.zip')
    local_zip = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_zip, 'r')
    zip_ref.extractall('tmp/')
    zip_ref.close()

    TRAINING_DIR = "tmp/rps/"

    train_datagen = ImageDataGenerator(
        # YOUR CODE HERE)
        rescale=1/255.0,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.1,)

    # YOUR CODE HERE
    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        batch_size=64,
        class_mode='categorical',
        subset='training'
    )

    valid_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        batch_size=128,
        subset='validation'
    )

    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (2, 2), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu', padding="same"),
        MaxPooling2D(2, 2),
        Conv2D(256, (6, 6), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
        Dense(3, activation='softmax')
    ])

    model.summary()

    checkpoint_path = 'rps_checkpoint.ckpt'

    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 monitor='val_loss',
                                 verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  patience=2,
                                  factor=0.5,
                                  verbose=1)

    earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=3, verbose=1)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    model.fit(train_generator,
              steps_per_epoch=len(train_generator),
              validation_data=(valid_generator),
              validation_steps=len(valid_generator),
              epochs=21,
              callbacks=[checkpoint, reduce_lr, earlystopping],
              verbose=1)

    model.load_weights(checkpoint_path)

    return model


# Note that you'll need to save your model as a .h5 like this
# This .h5 will be uploaded to the testing infrastructure
# and a score will be returned to you
if __name__ == '__main__':
    model = solution_model()
    model.save("rps_model.h5")
