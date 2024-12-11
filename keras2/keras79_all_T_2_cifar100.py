from tensorflow.keras.applications import VGG16, VGG19 #
from tensorflow.keras.applications import ResNet50, ResNet50V2
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import DenseNet201, DenseNet121, DenseNet169
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile, NASNetLarge
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7
from tensorflow.keras.applications import Xception

#10개보다 더 해도 괜춘!
#shape 오류인거는 내용 명시하고 다른 모델 쓸 것! 

'''
01. VGG19
02. Xception x
03. ResNet50
04. ResNet101
05. InceptionV3 x
06. InceptionResNetV3 x
07. DenseNet121 x
08. MobileNetV2
09. NasNetMobile
10. EfficientNetB0


'''

model_list = [VGG19, Xception, ResNet50, ResNet101, InceptionV3, DenseNet121, DenseNet121, MobileNetV2, NASNetMobile, EfficientNetB0 ]


# GAP 써라!
# 기존거와 최고 성능 비교

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10, cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data() 

# TensorFlow를 사용하여 일괄적으로 리사이즈
x_train = tf.image.resize(x_train, (75, 75)).numpy()
x_test = tf.image.resize(x_test, (75, 75)).numpy()

x_train = x_train/255.
x_test = x_test/255.

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
# y_train = y_train.reshape(-1, 1)
# y_test = y_test.reshape(-1, 1)
y_train = ohe.fit_transform(y_train)
y_test = ohe.transform(y_test)

for i in model_list:

    models = i(#weights='imagenet',
                include_top=False,          
                input_shape=(75, 75, 3))    

    models.trainable=False

    model = Sequential()
    model.add(models)
    model.add(GlobalAveragePooling2D())
    # model.add(Flatten())
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(100, activation='softmax'))

    # model.summary()


    # 3. 컴파일 훈련
    model.compile(loss = 'categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

    start_time = time.time()

    es = EarlyStopping(
        monitor = 'val_loss',
        mode = 'min',
        verbose=0,
        patience=20,
        restore_best_weights=True
    )

    import datetime
    date = datetime.datetime.now()
    date = date.strftime('%m%d_%H%M')


    model.fit(x_train, y_train, epochs = 1000, batch_size = 128, verbose=0, validation_split=0.25, callbacks=[es])
    end_time = time.time()

    #평가 예측
    loss = model.evaluate(x_test, y_test)
    y_predict = model.predict(x_test)

    # y_test = np.argmax(y_test, axis=1).reshape(-1, 1)
    # y_predict = np.argmax(y_predict, axis=1).reshape(-1, 1)

    # acc = accuracy_score(y_test, y_predict)
    print('모델이름 :', models.name)
    print("로스값 : ", loss[0])
    print("정확도 : ", loss[1])
    print('----------------------------')

    # 가중치 동결
    # 로스값 :  2.6119139194488525
    # 정확도 :  0.12890000641345978


