# 데이터 모양 출력
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train:%s y_train:%s x_test:%s, y_test:%s" %(
    x_train.shape, y_train.shape, x_test.shape, y_test.shape
))

# 학습데이터 그림 그려서 출력하기
import matplotlib.pyplot as plt

plt.figure()
plt.imshow(x_train[0])
plt.show()

# 해당 그림의 픽셀값 출력하기
for y in range(28) :
  for x in range(28) :
    print("%4s" %x_train[0][y][x], end=' ')
  print()

# 학습데이터 그림 그려서 출력하기
for i in range(25) :
  plt.subplot(5,5,i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(x_train[i], cmap=plt.cm.binary)
  plt.xlabel(y_train[i])
plt.show()

# 인공신경망 학습하기
x_train, x_test = x_train/255.0, x_test/255.0
x_train, x_test = x_train.reshape(60000, 784), x_test.reshape(10000, 784)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

p_test = model.predict(x_test)
print('p_test[0]:', p_test[0])

import numpy as np
print('p_test[0]:', p_test[0], 'y_test[0]:', y_test[0])