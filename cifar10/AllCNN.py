#Implementing the all-CNN architecture specified in the Striving for Simplicity paper.
import numpy as np 
import scipy.ndimage
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Convolution2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras import optimizers
import keras.backend as K
from keras import initializers

def scheduler(epoch):
    if epoch >= 200 and epoch%50==0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr*0.1)
        print("lr changed to {}".format(lr*0.1))
    return K.get_value(model.optimizer.lr)

#Download the cifar10 dataset
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

init = initializers.random_normal(stddev=0.05)

model = Sequential()
# model.add(Dropout(0.2, input_shape=(32, 32, 3)))
model.add(Convolution2D(96, 3, 3, border_mode = 'same', activation = 'relu', input_shape=(32, 32, 3)))
model.add(Convolution2D(96, 3, 3, border_mode = 'same', activation = 'relu'))
model.add(Convolution2D(96, 3, 3, subsample=(2, 2), border_mode = 'same', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(192, 3, 3, border_mode = 'same', activation = 'relu'))
model.add(Convolution2D(192, 3, 3, border_mode = 'same', activation = 'relu'))
model.add(Convolution2D(192, 3, 3, subsample=(2, 2), border_mode = 'same', activation = 'relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(192, 3, 3, border_mode = 'same', activation = 'relu'))
model.add(Convolution2D(192, 1, 1, border_mode = 'valid', activation = 'relu'))
model.add(Convolution2D(10, 1, 1, border_mode = 'valid', activation = 'relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(10,activation = 'softmax'))

model.summary()
lr_decay = LearningRateScheduler(scheduler)

sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)

model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

# model_history = model.fit(X_train, Y_train, batch_size = 128, epochs = 200,
# 	validation_data = (X_test, Y_test),
# 	callbacks=[lr_decay],
# 	verbose = 1)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False) 

datagen.fit(X_train)
model_history = model.fit_generator(datagen.flow(X_train, Y_train, 
	batch_size = 32),
	samples_per_epoch = X_train.shape[0],
	epochs = 350, 
	validation_data = (X_test, Y_test),
	callbacks=[lr_decay], 
	verbose = 1)

print(model_history.history.keys())

plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('AllCNN_loss1.png')

plt.plot(model_history.history['acc'])
plt.plot(model_history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('AllCNN_accuracy1.png')


model.save_weights('../AllCNN1.h5')


