import tensorflow as tf

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train:%s y_train:%s x_test:%s y_test:%s" %(
    x_train.shape, y_train.shape, x_test.shape, y_train.shape))

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(x_train[0])
plt.show()

for y in range(28) :
    for x in range(28) :
        print("%4s" %x_train[0][y][x], end='')
    print()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()

x_train, x_test = x_train/255.0, x_test/255.0
x_train, x_test = x_train.reshape(60000, 784), x_test.reshape(10000, 784)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

p_test=model.predict(x_test)
print('p_test:', p_test[0])

import numpy as np

print('p_test[0]:', np.argmax(p_test[0]), 'y_test[0]:', y_test[0])

# 첫번째 테스트 이미지 구조를 다시 바꿔서 출력해본다.
x_test = x_test.reshape(10000, 28, 28)
plt.imshow(x_test[0])
plt.show()

plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(p_test[i])])
plt.show()

cnt_wrong = 0
p_wrong = []
for i in range(10000) :
    if np.argmax(p_test[i]) != y_test[i] :
        cnt_wrong += 1
        p_wrong.append(i)
print("cnt_wrong: ", cnt_wrong)
print("p_wrong 10: ", p_wrong[:10])

plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[p_wrong[i]], cmap=plt.cm.binary)
    plt.xlabel("%s p%s y%s" %(p_wrong[i], class_names[np.argmax(p_test[p_wrong[i]])], class_names[y_test[p_wrong[i]]]))
plt.show()

x = 2
t = 10
w = 3
b = 1

y = x*w + 1*b
print('y=%6.3f' %y)

yb = y-t
xb = yb*w
wb = yb*x
bb = yb*b
print('xb=%6.3f wb=%6.3f bb=%6.3f' %(xb, wb, bb))

lr=0.01
w=w-(wb*lr)
b=b-(bb-lr)
print('x=%6.3f w=%6.3f b=%6.3f' %(x, w, b))


#약 200번의 반복으로 값만 입력해서 y=3x+1이라는 함수식을 만들 수 있다.
xs = [-1.,0.,1.,2.,3.,4.]
ys = [-2.,1.,4.,7.,10.,13.]
w = 10.
b = 10.

for epoch in range(2000) :
    for n in range(6) :
        y = xs[n]*w + b*1

        #오차율 구하기
        t=ys[n]
        E=(y-t)**2/2


        #역전파 구하기
        yb = y-t
        wb = yb*xs[n]
        bb = yb*1

        #학습시키기.
        lr=0.01
        w = w-lr*wb
        b = b-lr*bb
        if epoch%200==1 # import tensorflow as tf

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train:%s y_train:%s x_test:%s y_test:%s" %(
    x_train.shape, y_train.shape, x_test.shape, y_train.shape))

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(x_train[0])
plt.show()

for y in range(28) :
    for x in range(28) :
        print("%4s" %x_train[0][y][x], end='')
    print()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()

x_train, x_test = x_train/255.0, x_test/255.0
x_train, x_test = x_train.reshape(60000, 784), x_test.reshape(10000, 784)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)

p_test=model.predict(x_test)
print('p_test:', p_test[0])

import numpy as np

print('p_test[0]:', np.argmax(p_test[0]), 'y_test[0]:', y_test[0])

# 첫번째 테스트 이미지 구조를 다시 바꿔서 출력해본다.
x_test = x_test.reshape(10000, 28, 28)
plt.imshow(x_test[0])
plt.show()

plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(p_test[i])])
plt.show()

cnt_wrong = 0
p_wrong = []
for i in range(10000) :
    if np.argmax(p_test[i]) != y_test[i] :
        cnt_wrong += 1
        p_wrong.append(i)
print("cnt_wrong: ", cnt_wrong)
print("p_wrong 10: ", p_wrong[:10])

plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[p_wrong[i]], cmap=plt.cm.binary)
    plt.xlabel("%s p%s y%s" %(p_wrong[i], class_names[np.argmax(p_test[p_wrong[i]])], class_names[y_test[p_wrong[i]]]))
plt.show()

x = 2
t = 10
w = 3
b = 1

y = x*w + 1*b
print('y=%6.3f' %y)

yb = y-t
xb = yb*w
wb = yb*x
bb = yb*b
print('xb=%6.3f wb=%6.3f bb=%6.3f' %(xb, wb, bb))

lr=0.01
w=w-(wb*lr)
b=b-(bb-lr)
print('x=%6.3f w=%6.3f b=%6.3f' %(x, w, b))


#약 200번의 반복으로 값만 입력해서 y=3x+1이라는 함수식을 만들 수 있다.
xs = [-1.,0.,1.,2.,3.,4.]
ys = [-2.,1.,4.,7.,10.,13.]
w = 10.
b = 10.

for epoch in range(2000) :
    for n in range(6) :
        y = xs[n]*w + b*1

        #오차율 구하기
        t=ys[n]
        E=(y-t)**2/2


        #역전파 구하기
        yb = y-t
        wb = yb*xs[n]
        bb = yb*1

        #학습시키기.
        lr=0.01
        w = w-lr*wb
        b = b-lr*bb
        if epoch%200==1 and n==0:
            print("w:%6.3f, b:%6.3f" %(w,b))


xs = [-1.,0.,1.,2.,3.,4.]
ys = [-2.,1.,4.,7.,10.,13.]
w = 10.
b = 10.

for epoch in range(2000) :
    for n in range(6) :
        y = xs[n]*w + b*1

        #오차율 구하기
        t=ys[n]
        E=(y-t)**2/2


        #역전파 구하기
        yb = y-t
        wb = yb*xs[n]
        bb = yb*1

        #학습시키기
        lr=0.01
        w = w-lr*wb
        b = b-lr*bb
        if epoch%200==1 and n==0:
            print("w:%6.3f, b:%6.3f" %(w,b))


import tensorflow as tf
import numpy as np

xs = np.array([-1.,0.,1.,2.,3.,4.])
ys = np.array([-2.,1.,4.,7.,10.,13.])

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(xs, ys, epochs=5)

p = model.predict([10.0])

print('p: ', p)

import time
import matplotlib.pyplot as plt
NUM_SAMPLES = 1000

np.random.seed(int(time.time()))
xs = np.random.uniform(-2, 0.5, NUM_SAMPLES)
np.random.shuffle(xs)

ys = 2*xs**2 + 3*xs + 5
plt.plot(xs, ys, 'b.')
plt.show()

ys += 0.1 * np.random.randn(NUM_SAMPLES)
plt.plot(xs, ys, 'g.')
plt.show()

NUM_SPLIT = int(0.8*NUM_SAMPLES)

x_train, x_test = np.split(xs, [NUM_SPLIT])
y_train, y_test = np.split(ys, [NUM_SPLIT])

plt.plot(x_train, y_train, 'b.', label='train')
plt.plot(x_test, y_test, 'r.', label='test')
plt.legend()
plt.show()

import tensorflow as tf

model_f = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model_f.compile(optimizer='rmsprop', loss='mse')
model_f.fit(x_train, y_train, epochs=600)


p_test = model_f.predict(x_test)

plt.plot(x_train, y_train, 'b.', label='train')
plt.plot(x_test, p_test, 'r.', label='test')
plt.legend()
plt.show()

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train, x_test = x_train.reshape((60000, 784)), x_test.reshape((10000, 784))

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train:%s, y_train:%s, x_test:%s, y_test:%s" %(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(x_train[0])
plt.show()

for y in range(28) :
    for x in range(28) :
        print("%4s" %x_test[0][y][x], end='')
    print()

plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(y_test[i])
plt.show()

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
print("p_test[0]:", np.argmax(p_test[0]), "y_test[0]:", y_test[0])

x_test = x_test.reshape(10000, 28, 28)

plt.figure()
plt.imshow(x_test[0])
plt.show()


plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(np.argmax(p_test[i]))
plt.show()


cnt_wrong=0
p_wrong=[]
for i in range(10000) :
    if np.argmax(p_test[i]) != y_test[i] :
        cnt_wrong+=1
        p_wrong.append(i)


plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[p_wrong[i]], cmap=plt.cm.binary)
    plt.xlabel("%s p:%s y:%s" %(p_wrong[i], np.argmax(p_test[p_wrong[i]]), y_test[p_wrong[i]]))
plt.show()
and n==0:
            print("w:%6.3f, b:%6.3f" %(w,b))


xs = [-1.,0.,1.,2.,3.,4.]
ys = [-2.,1.,4.,7.,10.,13.]
w = 10.
b = 10.

for epoch in range(2000) :
    for n in range(6) :
        y = xs[n]*w + b*1

        #오차율 구하기
        t=ys[n]
        E=(y-t)**2/2


        #역전파 구하기
        yb = y-t
        wb = yb*xs[n]
        bb = yb*1

        #학습시키기
        lr=0.01
        w = w-lr*wb
        b = b-lr*bb
        if epoch%200==1 and n==0:
            print("w:%6.3f, b:%6.3f" %(w,b))


import tensorflow as tf
import numpy as np

xs = np.array([-1.,0.,1.,2.,3.,4.])
ys = np.array([-2.,1.,4.,7.,10.,13.])

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(xs, ys, epochs=5)

p = model.predict([10.0])

print('p: ', p)

import time
import matplotlib.pyplot as plt
NUM_SAMPLES = 1000

np.random.seed(int(time.time()))
xs = np.random.uniform(-2, 0.5, NUM_SAMPLES)
np.random.shuffle(xs)

ys = 2*xs**2 + 3*xs + 5
plt.plot(xs, ys, 'b.')
plt.show()

ys += 0.1 * np.random.randn(NUM_SAMPLES)
plt.plot(xs, ys, 'g.')
plt.show()

NUM_SPLIT = int(0.8*NUM_SAMPLES)

x_train, x_test = np.split(xs, [NUM_SPLIT])
y_train, y_test = np.split(ys, [NUM_SPLIT])

plt.plot(x_train, y_train, 'b.', label='train')
plt.plot(x_test, y_test, 'r.', label='test')
plt.legend()
plt.show()

import tensorflow as tf

model_f = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(1,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)
])

model_f.compile(optimizer='rmsprop', loss='mse')
model_f.fit(x_train, y_train, epochs=600)


p_test = model_f.predict(x_test)

plt.plot(x_train, y_train, 'b.', label='train')
plt.plot(x_test, p_test, 'r.', label='test')
plt.legend()
plt.show()

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train, x_test = x_train.reshape((60000, 784)), x_test.reshape((10000, 784))

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train:%s, y_train:%s, x_test:%s, y_test:%s" %(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(x_train[0])
plt.show()

for y in range(28) :
    for x in range(28) :
        print("%4s" %x_test[0][y][x], end='')
    print()

plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(y_test[i])
plt.show()

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
print("p_test[0]:", np.argmax(p_test[0]), "y_test[0]:", y_test[0])

x_test = x_test.reshape(10000, 28, 28)

plt.figure()
plt.imshow(x_test[0])
plt.show()


plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[i], cmap=plt.cm.binary)
    plt.xlabel(np.argmax(p_test[i]))
plt.show()


cnt_wrong=0
p_wrong=[]
for i in range(10000) :
    if np.argmax(p_test[i]) != y_test[i] :
        cnt_wrong+=1
        p_wrong.append(i)


plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[p_wrong[i]], cmap=plt.cm.binary)
    plt.xlabel("%s p:%s y:%s" %(p_wrong[i], np.argmax(p_test[p_wrong[i]]), y_test[p_wrong[i]]))
plt.show()


x1, x2 = 2, 3
t=27
w1=3
w2=4
b=1

for epoch in range(2000) :
    print('epoch=%d' %epoch)

    y=x1*w1+x2*w2+1*b
    print('y=%6.3f'%y)

    E=(y-t)**2/2
    print('E=%.7f'%E)
    if E<0.0000001:
        break

    yb=y-t
    x1b,x2b=yb*w1, yb*w2
    w1b=yb*x1
    w2b=yb*x2
    bb=yb*1
    print('x1b, x2b = %6.3f, %6.3f'%(x1b, x2b))
    print('w1b, w2b, bb = %6.3f, %6.3f, %6.3f' %(w1b, w2b, bb))

    lr=0.01
    w1 = w1 - lr*w1b
    w2 = w2 - lr*w2b
    b = b - lr*bb
    print('w1, w2, b = %6.3f, %6.3f, %6.3f'%(w1, w2, b))

x1, x2 = 2, 3
w1, w2, w3, w4 = 3, 4, 5, 6
b1, b2 = 1, 2
t1, t2 = 27, -30

for epoch in range(2000) :
    print('epoch=%d'%epoch)

    y1=x1*w1+x2*w2+b1*1
    y2=x1*w3+x2*w4+b2*1

    E=(y1-t1)**2/2 + (y2-t2)**2/2
    print('E:%.7f'%E)
    if E < 0.0000001 :
        break

    y1b = y1-t1
    y2b = y2-t2
    x1b = y1b*w1+y2b*w3
    x2b = y1b*w2+y2b*w4
    w1b = y1b*x1
    w2b = y1b*x2
    w3b = y2b*x1
    w4b = y2b*x2
    b1b, b2b = y1b*1, y2b*1

    lr = 0.01
    w1,w3 = w1-w1b*lr, w3-w3b*lr
    w2, w4 = w2-w2b*lr, w4-w4b*lr
    b1, b2 = b1-lr*b1b, b2-lr*b2b

x1, x2, x3 = 2,3,4
t1, t2, t3 = 27, -30, 179
w1, w4, w7 = 3,5,8
w2, w5, w8 = 4,6,9
w3, w6, w9 = 5,7,10
b1, b2, b3 = 1,2,3

for epoch in range(2000) :
    print('epoch=%d'%epoch)

    y1=x1*w1+x2*w2+x3*w3+b1*1
    y2=x1*w4+x2*w5+x3*w6+b2*1
    y3=x1*w7+x2*w8+x3*w9+b3*1

    E=(y1-t1)**2/2+(y2-t2)**2/2+(y3-t3)**2/2
    print('E:%.7f'%E)
    if E<0.0000001 :
        break

    y1b,y2b,y3b=y1-t1, y2-t2, y3-t3
    w1b=y1b*x1
    w2b=y1b*x2
    w3b=y1b*x3
    w4b=y2b*x1
    w5b=y2b*x2
    w6b=y2b*x3
    w7b=y3b*x1
    w8b=y3b*x2
    w9b=y3b*x3
    b1b=y1b
    b2b=y2b
    b3b=y3b

    lr=0.01
    w1 = w1 - lr * w1b
    w2 = w2 - lr * w2b
    w3 = w3 - lr * w3b
    w4 = w4 - lr * w4b
    w5 = w5 - lr * w5b
    w6 = w6 - lr * w6b
    w7 = w7 - lr * w7b
    w8 = w8 - lr * w8b
    w9 = w9 - lr * w9b
    b1, b2, b3 = b1 - lr * b1b, b2 - lr * b2b, b3 - lr * b3b


i1, i2 = .05, .10
t1, t2 = .01, .99

w1, w3 = .15, .25
w2, w4 = .20, .30
b1, b2 = .35, .35

w5, w7 = .40, .50
w6, w8 = .45, .55
b3, b4 = .60, .60

for epoch in range(2000) :
    print('Epoch:%d'%epoch)

    h1 = i1 * w1 + i2 * w2 + b1 * 1
    h2 = i1 * w3 + i2 * w4 + b2 * 1
    o1 = h1 * w5 + h2 * w6 + b3 * 1
    o2 = h1 * w7 + h2 * w8 + b4 * 1

    E = (o1-t1)**2/2 + (o2-t2)**2/2
    print('E:%.7f'%E)
    if E < 0.0000001 :
        break

    o1b, o2b = o1 - t1, o2 - t2
    h1b = o1b * w5 + o2b * w7
    h2b = o1b * w6 + o2b * w8
    w1b, w3b = h1b * i1, h2b * i1
    w2b, w4b = h1b * i2, h2b * i2
    b1b, b2b = h1b * 1, h2b * 1
    w5b, w7b = h1 * o1b, h1 * o2b
    w6b, w8b = h2 * o1b, h2 * o2b
    b3b, b4b = o1b * 1, o2b * 1

    lr = 0.01
    w1 = w1 - lr * w1b
    w2 = w2 - lr * w2b
    w3 = w3 - lr * w3b
    w4 = w4 - lr * w4b
    w5 = w5 - lr * w5b
    w6 = w6 - lr * w6b
    w7 = w7 - lr * w7b
    w8 = w8 - lr * w8b
    b1 = b1 - lr * b1b
    b2 = b2 - lr * b2b
    b3 = b3 - lr * b3b
    b4 = b4 - lr * b4b


