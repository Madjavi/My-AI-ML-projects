import tensorflow as tf
import time as tm
import numpy as npy
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model


# This class starts the algorith.
class Start:
    # Tensorflow version.
    ver = tf.__version__
    msg_0_0 = f"\nImplementation of CIFAR10 with CNNs Using TensorFlow version: {ver}\n"
    print(msg_0_0)
    tm.sleep(1)
    # Progress bar.
    for progress in tqdm(range(101), desc="Loading..", ascii=False, ncols=75):
        tm.sleep(0.01)

    print("\nStarting----->\n")
    tm.sleep(1)


# This class loads, reads, and split the CIFAR10 dataset.
class dataset_load:
    # CIFAR10 dataset.
    dataset_cif10 = tf.keras.datasets.cifar10
    # Loads, reads, and split the dataset.
    (x_train, y_train), (x_test, y_test) = dataset_cif10.load_data()
    msg_0 = x_train.shape, y_train.shape, x_test.shape, y_test.shape
    # Dataset progress bar.
    for progress_2 in tqdm(range(101), desc="Loading Dataset", ascii=False, ncols=75):
        tm.sleep(0.01)

    msg_1 = f"\nCIFAR10 dataset--------------------------------------->>\n{msg_0}\n\n"
    print(msg_1)
    tm.sleep(1)


# This class normalizes the test data.
class pre_processing:
    # Splits and normalises the dataset.
    dataset_load.x_train, dataset_load.x_test = dataset_load.x_train / 255.0, dataset_load.x_test / 255.0
    dataset_load.y_train, dataset_load.y_test = dataset_load.y_train.flatten(), dataset_load.y_test.flatten()


# This class verifies the dataset classification.
class data_verification_Val:
    dataset_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure(figsize=(10, 10))
    for labels in range(25):
        plt.subplot(5, 5, labels + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(dataset_load.x_train[labels])
        plt.xlabel(dataset_labels[dataset_load.y_train[labels]])
    plt.show()


# This class houses the Convolution Neural Network (CNN).
class CNN:
    # CNN defined.
    CIFAR10_Model = tf.keras.Sequential()
    CIFAR10_Model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    CIFAR10_Model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    CIFAR10_Model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    CIFAR10_Model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    CIFAR10_Model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    CIFAR10_Model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    CIFAR10_Model.add(tf.keras.layers.Flatten())
    CIFAR10_Model.add(tf.keras.layers.Dropout(0.2))
    CIFAR10_Model.add(tf.keras.layers.Dense(512, activation='relu'))
    CIFAR10_Model.add(tf.keras.layers.Dropout(0.2))
    CIFAR10_Model.add(tf.keras.layers.Dense(10))

    # CNN model summary.
    CIFAR10_Model.summary()

    # Compile the model and defined loss function.
    CIFAR10_Model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['Accuracy'])

    # Training and validation.
    Training = CIFAR10_Model.fit(dataset_load.x_train, dataset_load.y_train, epochs=10,
                                 validation_data=(dataset_load.x_test, dataset_load.y_test))
    # Plost the model overall accuracy and loss.
    plt.plot(Training.history['Accuracy'], label='Acc', color='red')
    plt.plot(Training.history['val_Accuracy'], label='val_Acc', color='green')
    plt.title('CIFAR_10 Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

    # Loss and accuracy variables for evaluation.
    loss, accuracy = CIFAR10_Model.evaluate(dataset_load.x_test, dataset_load.y_test, verbose=2)
    msg_2 = f"\nCIFAR10 Classification Model overall accuracy: {accuracy}"
    msg_3 = f"\nCIFAR10 Model overall loss: {loss}\n"
    print(msg_2 + msg_3)


# This class repeats the training process with different hyperparameters to decrease the loss.
class decrease_Loss:
    # Number of bach sizes.
    batch_size = 32
    # The image data generator will label based on the number of subdirectories.
    generate_data = tf.keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1,
                                                                    horizontal_flip=True)
    train_gen = generate_data.flow(dataset_load.x_train, dataset_load.y_train, batch_size=batch_size)
    Epochs = dataset_load.x_train.shape[0] // batch_size
    # The training #2 will re feed the test set to the cnn with the new hyperparameters.
    Training_2 = CNN.CIFAR10_Model.fit(train_gen, validation_data=(dataset_load.x_test, dataset_load.y_test),
                                       steps_per_epoch=Epochs, epochs=50)
    # The plot will show the new accuracy number and loss per epochs.
    plt.plot(Training_2.history['Accuracy'], label='Acc', color='red')
    plt.plot(Training_2.history['val_Accuracy'], label='val_Acc', color='green')
    plt.title('Increase Accuracy & decrease Loss')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

    # New loss and accuracy variables for re-evaluation.
    loss_2, accuracy_2 = CNN.CIFAR10_Model.evaluate(dataset_load.x_test, dataset_load.y_test)
    msg_4 = f"\nNew loss: {loss_2}"
    msg_5 = f"\nNew accuracy: {accuracy_2}\n"
    print(msg_4 + msg_5)


# This  class shall make a prediction on the test data and generates a h5 file of the trained data.
class prediction:
    labels = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()
    selection = 0
    plt.imshow(dataset_load.x_test[selection])
    img_load = npy.array(dataset_load.x_test[selection])
    re_shape = img_load.reshape(1, 32, 32, 3)
    prediction = labels[CNN.CIFAR10_Model.predict(re_shape).argmax()]
    o_labels = labels[dataset_load.y_test[selection]]
    plt.show()

    predicted_img = f"\nOriginal image is a {o_labels} and the predicted label is {prediction}\n"
    print(predicted_img)
    tm.sleep(1)

    for progress_2 in tqdm(range(101), desc="Saving h5 file:", ascii=False, ncols=75):
        tm.sleep(0.01)

        h5_file_sever = CNN.CIFAR10_Model.save('CIFAR10 with CNNs Using TensorFlow.h5')

    msg_1 = f"\nh5 trained model file has been saved!"
    print(msg_1)


def main():
    Start()
    dataset_load()
    data_verification_Val()
    CNN()
    decrease_Loss()
    prediction()


if __name__ == "__main__":
    main()
