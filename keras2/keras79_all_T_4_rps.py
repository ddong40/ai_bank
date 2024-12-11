import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

tf.random.set_seed(333)
np.random.seed(333)
print(tf.__version__)   # 2.7.4

train_datagen = ImageDataGenerator(
    rescale=1./255,
   
)

test_datagen = ImageDataGenerator(
    rescale=1./255,             # test 데이터는 rescale만 하고 절대 건들지 않는다!!
)

path_train = './_data/image/rps'
# path_test = './_data/image/brain/test/'

xy_train = train_datagen.flow_from_directory(       # 디렉토리로 부터 흘러가듯 가져오세요
    path_train,
    target_size=(100, 100),                         # 이미지 사이즈 통일
    batch_size=10000,                                  # (10,200,200,1)
    class_mode='categorical',                     # 다중분류 - 원핫도 되서 나와욤
    # class_mode='sparse',                            # 다중분류
    # class_mode='binary',                            # 이진분류
    # class_mode='none',                            # y값 없다!
    
    # color_mode='grayscale',
    color_mode='rgb',

    shuffle=True,
)   # Found 160 images belonging to 2 classes

# print(xy_train[0][1])

# print(xy_train[0][0].shape)


y = xy_train[0][1]

x_train, x_test, y_train, y_test = train_test_split(
            xy_train[0][0],
            y,
            train_size=0.8,
            stratify=y,
            random_state=9843
)


from tensorflow.keras.applications import VGG16

vgg16 = VGG16(#weights='imagenet',
              include_top=False,          
              input_shape=(100, 100, 3))    

vgg16.trainable=False

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(3, activation='softmax'))

model.summary()

####### [실습] ########
# 비교할것
# 1. 이전에 본인이 한 최상의 결과와
# 2. 가중치를 동결하지 않고 훈련시켰을때, trainable=Ture,(디폴트)
# 3. 가중치를 동결하고 훈련시켰을때, trainable=False 
#### 위에 2,3번할때는 time 체크할 것  

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
start = time.time()
es = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=15,
    restore_best_weights=True
)

model.fit(x_train, y_train, epochs=100, batch_size=64,
          verbose=1,
          validation_split=0.1,
          callbacks=[es]
          )
end = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test, verbose=1)
print('loss :',loss[0])
print('acc :',round(loss[1],4))

y_pre = model.predict(x_test)

y_pre = np.argmax(y_pre, axis=1).reshape(-1,1)
y_test = np.argmax(y_test, axis=1).reshape(-1,1)

accuracy_score = accuracy_score(y_test,np.round(y_pre))
print('acc_score :', accuracy_score)


# loss : 0.013397577218711376
# acc : 0.998
# acc_score : 0.998015873015873

# True
# loss : 1.0425083637237549
# acc : 0.4405
# acc_score : 0.44047619047619047

# False
# loss : 1.2337721273070201e-05
# acc : 1.0
# acc_score : 1.0