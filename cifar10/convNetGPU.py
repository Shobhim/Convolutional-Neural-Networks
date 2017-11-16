#A simple Convolutional Neural Network on Cifar-10 dataset.
import numpy as np 
import scipy.ndimage
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, GlobalMaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from keras.datasets import cifar10
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
num_train, img_channels, img_rows, img_cols =  X_train.shape
num_test, _, _, _ =  X_train.shape
num_classes = len(np.unique(Y_train))

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# convert class labels to binary class labels
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)

model = Sequential()
model.add(Convolution2D(48, 3, 3, border_mode = 'same', activation = 'relu', input_shape = (32, 32, 3)))
model.add(Convolution2D(48, 3, 3, border_mode = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))
model.add(Convolution2D(96, 3, 3, border_mode = 'same', activation = 'relu'))
model.add(Convolution2D(96, 3, 3, border_mode = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.5))
model.add(Convolution2D(192, 3, 3, border_mode = 'same', activation = 'relu'))
model.add(Convolution2D(192, 3, 3, border_mode = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


datagen = ImageDataGenerator(zoom_range=0.2, 
							 horizontal_flip=True)

model_history = model.fit_generator(datagen.flow(X_train, Y_train, 
	batch_size = 128),
	samples_per_epoch = X_train.shape[0],
	epochs = 200, 
	validation_data = (X_test, Y_test), 
	# nb_val_samples = X_test.shape[0],
	verbose = 1)

print(model_history.history.keys())

plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('convNetGPU.png')

#Save the weights
model.save_weights('../convNetGPU.h5')

