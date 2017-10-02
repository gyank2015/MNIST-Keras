from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import np_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(10)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
m_train = X_train.shape[0]
m_test = X_test.shape[0]
n = X_train.shape[1]*X_train.shape[2]
nb_class = 10
X_train = X_train.reshape(m_train,n).astype('float32')
X_test = X_test.reshape(m_test,n).astype('float32')
X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train, nb_class)
y_test = np_utils.to_categorical(y_test, nb_class)

model = Sequential()
model.add(Dense(1000, input_dim=n, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(nb_class, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=200, verbose=2, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print 'Accuracy: %0.2f%%'%(score[1]*100)

fig1 = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()
fig1.savefig('accuracy1.png')

fig2 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.grid()
plt.show()
fig2.savefig('loss1.png')