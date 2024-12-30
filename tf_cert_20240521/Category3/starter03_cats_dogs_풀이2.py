import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint

def preprocess(features):
    # YOUR CODE HERE
    image, label = tf.cast(features['image'], tf.float32) / 255.0, tf.one_hot(features['label'], 2)
    image = tf.image.resize(image, size=(224, 224))
    return image, label

dataset_name = 'cats_vs_dogs'
train_dataset, info = tfds.load(name=dataset_name, split="train[:20000]", with_info=True)
valid_dataset = tfds.load(name=dataset_name, split="train[20000:]")

train_dataset = train_dataset.repeat().map(preprocess).batch(32)
valid_dataset = valid_dataset.repeat().map(preprocess).batch(32)

total_size = 20000
steps_per_epoch = total_size // 32 + 1

total_valid_size = 3262
validation_steps = total_valid_size // 32 + 1


def solution_model():
    # model = # YOUR CODE HERE, BUT MAKE SURE YOUR LAST LAYER HAS 2 NEURONS ACTIVATED BY SOFTMAX
    #     tf.keras.layers.Dense(2, activation='softmax')
    # ])
    model = Sequential([
        Conv2D(filters=256, kernel_size=(3, 3), input_shape=(224, 224, 3), padding='same', activation='relu'),
        # shape (224, 224, 256)
        MaxPooling2D(2, 2),
        # shape (112, 112, 256)
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        # shape (112, 112, 128)
        MaxPooling2D(2, 2),
        # (56, 56, 128)
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        # (28, 28, 128)
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        # (14, 14, 256)
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(2, 2),
        # (7, 7, 256)
        Flatten(),
        Dense(256, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.00005), loss='categorical_crossentropy',
                  metrics=['acc'])

    checkpoint_path = 'cats_dogs_checkpoint_0921.ckpt'

    checkpoint = ModelCheckpoint(checkpoint_path,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 monitor='val_loss',
                                 verbose=1,
                                 )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     mode='min',
                                                     patience=2,
                                                     factor=0.5)

    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=3,
                                                     verbose=1)

    model.fit(train_dataset,
              steps_per_epoch=steps_per_epoch,
              epochs=10,
              validation_data=(valid_dataset),
              validation_steps=validation_steps,
              callbacks=[checkpoint, reduce_lr, earlystopping],
              verbose=2
              )

    model.load_weights(checkpoint_path)

    return model

if __name__ == '__main__':
    model = solution_model()
    model.save("cats_dogs_model_0921.h5")