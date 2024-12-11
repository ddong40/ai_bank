import tensorflow as tf
print(tf.__version__)

gpus = tf.config.experimental.list_physical_devices('GPU') #gpu가 어디있는지 현재
print(gpus)