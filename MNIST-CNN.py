import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(10)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
m_train = X_train.shape[0]
m_test = X_test.shape[0]
n = X_train.shape[1]
nb_class = 10
X_train = X_train.reshape(m_train,n,n,1).astype('float32')
X_test = X_test.reshape(m_test,n,n,1).astype('float32')
X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train, nb_class)
y_test = np_utils.to_categorical(y_test, nb_class)

model = Sequential()
model.add(Conv2D(32, (5, 5), input_shape=(28, 28,1), activation='relu'))
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
#Fully Connected Layer
model.add(Dense(128, activation='relu'))
model.add(Dense(nb_class, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=200, verbose=2)

score = model.evaluate(X_test, y_test, verbose=0)
print 'Accuracy: %.2f%%'%(score[1]*100)

fig1 = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()
fig1.savefig('accuracy2.png')

fig2 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()
fig2.savefig('loss2.png')