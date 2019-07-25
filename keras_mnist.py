from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as k_backend

batch_size = 128
num_classes = 10
epochs = 2


def read_data():
    # Define input image dimensions
    img_rows, img_cols = 28, 28

    # Load the data, split between train and test sets
    (data_train, labels_train), (data_test, labels_test) = mnist.load_data()
    # For the ease of debugging - take a small portion of the data. Take entire collection when running on VM
    data_train = data_train[0:6000, :, :]
    labels_train = labels_train[0:6000]
    data_test = data_test[0:1000, :, :]
    labels_test = labels_test[0:1000]

    if k_backend.image_data_format() == 'channels_first':
        data_train = data_train.reshape(data_train.shape[0], 1, img_rows, img_cols)
        data_test = data_test.reshape(data_test.shape[0], 1, img_rows, img_cols)
        data_shape = (1, img_rows, img_cols)
    else:
        data_train = data_train.reshape(data_train.shape[0], img_rows, img_cols, 1)
        data_test = data_test.reshape(data_test.shape[0], img_rows, img_cols, 1)
        data_shape = (img_rows, img_cols, 1)

    data_train = data_train.astype('float32')
    data_test = data_test.astype('float32')
    data_train /= 255
    data_test /= 255
    print('data_train shape:', data_train.shape)
    print(data_train.shape[0], 'train samples')
    print(data_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    labels_train = keras.utils.to_categorical(labels_train, num_classes)
    labels_test = keras.utils.to_categorical(labels_test, num_classes)

    return data_train, data_test, labels_train, labels_test, data_shape


def main():
    x_train, x_test, y_train, y_test, input_shape = read_data()
    tb_callback = keras.callbacks.TensorBoard(log_dir='./TensorBoardLogs',
                                              histogram_freq=0, write_graph=True, write_images=True)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy', keras.losses.binary_crossentropy])

    # Consider another loss function

    # model.compile(loss=keras.losses.binary_crossentropy,
    #               optimizer=keras.optimizers.Adadelta(),
    #               metrics=[keras.losses.categorical_crossentropy, 'accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[tb_callback])
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    model.summary()
    model.save('myModel.h5')


if '__main__' == __name__:
    main()
