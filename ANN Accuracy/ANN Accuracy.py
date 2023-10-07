# Imported libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Option #2: Tensorflow ANN Model.

# Using Tensorflow and your own research, write a basic Tensorflow ANN model to perform a basic function of your
# choosing. Your submission should be inference-ready upon execution and include all model checkpoints necessary for
# inference. Your submission should include a self-executable Python script, in which model inference can be
# confirmed. The executable script should visually display results. Accuracy will not be graded but must run without
# error and display classification results on-screen.

# This class contains the main dataset.
class dataSet:
    data_main = "Simple Dataset.csv"


# This class has the user input function with the options.
class usr_input:
    message_main = "The Purpose of this ANN model is simply to measure its accuracy\n"
    message1 = '\n1. Input dataset'
    message2 = '\n2. Do nothing and exit\n'
    print(message_main, message1, message2)
    print('Enter Option: ', end='')
    user = int(input())

    if user == 1:
        print("\nLoading Dataset: Simple Dataset.csv................\n")

        d_set = pd.read_csv(dataSet.data_main)
        print(d_set.head())

    elif user == 2:
        print("Have a nice Day!")
        quit(0)


# This class contains the main brain of the entire operation; the artificial neural network with
# some other input functions.
class artificialNN:
    message = "\nDataset ready!\n"
    message2 = "Do you like to proceed with the training?"
    print(message, message2)
    print("yes or no: ", end='')
    userI = input()

    if userI == "yes":
        np.set_printoptions(precision=3, suppress=True)

        d_set_2 = pd.read_csv(dataSet.data_main)

        train, test = train_test_split(d_set_2, test_size=0.2, random_state=42, shuffle=True)

        x = np.column_stack((train.x.values, train.y.values))
        y = train.writing.values

        checkpoint_path = dataSet.data_main
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss",
                                                         save_weights_only=False, verbose=0, save_freq='epoch',
                                                         save_best_only=False)

        # ANN model
        model_training = keras.Sequential([
            keras.layers.Dense(16, input_shape=(2,), activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(2, activation='softmax')
        ])

        # model compiler
        model_training.compile(optimizer='adam',
                               loss=keras.losses.SparseCategoricalCrossentropy(),
                               metrics=['accuracy'])

        model_training.summary()

        # fitting and callback function.
        training_data = model_training.fit(x, y, validation_split=0.33, epochs=10, batch_size=8)
        print(cp_callback)

        # accuracy plot

        plt.plot(training_data.history['accuracy'])
        plt.plot(training_data.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        # value loss plot

        plt.plot(training_data.history['loss'], label='loss')
        plt.plot(training_data.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif userI == "no":
        print("In Case I Don't See Ya, Good Afternoon, Good Evening And Goodnight.")
        quit(0)


# main function defined.
def main():
    usr_input()


# Runs the main operation of the ANN.

if __name__ == "main":
    main()
