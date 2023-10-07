import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from os import system, name
import numpy as npy


# This class config houses all the variables for the clr_t() function.
class config:
    sys1 = 'nt'
    sys2 = 'clear'
    sys3 = 'cls'


# This function clears the console terminal.
def clr_T():
    if name == config.sys1:
        _ = system(config.sys3)
    else:
        _ = system(config.sys2)


# This class imports the dataset.
class sources:
    theSession = tf.compat.v1.InteractiveSession()
    mnist_model = tf.keras.datasets.mnist


# This class loads and pre-process the dataset to convert them into tensors.
class load_pre_pross:
    (x_train, y_train), (x_test, y_test) = sources.mnist_model.load_data()

    train_images = x_train.reshape(60000, 784)
    test_images = x_test.reshape(10000, 784)
    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    x_train, x_test = train_images / 255.0, test_images / 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)


# This function visualize the image in the MNIST dataset.
def dataset_samples(num):
    clr_T()
    print(load_pre_pross.y_train[num])

    label = load_pre_pross.y_train[num].argmax(axis=0)
    image = load_pre_pross.x_train[num].reshape([28, 28])
    plt.title('MNIST_Sample_Img: %d Label %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()


dataset_samples(1)


# This function visualizes how the data set is being feed to the model.
def data_visualization():
    images = load_pre_pross.x_train[0].reshape([1, 784])
    for data in range(1, 500):
        images = npy.concatenate((images, load_pre_pross.x_train[data].reshape([1, 784])))

    plt.title('MNIST DATASET')
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()


data_visualization()


# This class houses the neural network that is going to be trained.
class NN_model_trainer:
    tf.compat.v1.disable_eager_execution()

    input_images = tf.compat.v1.placeholder(tf.float32, shape=[None, 784])
    target_labels = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])

    hidden_neurons = 100
    input_weights = tf.Variable(tf.compat.v1.truncated_normal([784, hidden_neurons]))
    input_biases = tf.Variable(tf.zeros([hidden_neurons]))

    hidden_weights = tf.Variable(tf.compat.v1.truncated_normal([hidden_neurons, 10]))
    hidden_biases = tf.Variable(tf.zeros([10]))

    input_layer = tf.matmul(input_images, input_weights)
    hidden_layer = tf.nn.relu(input_layer + input_biases)
    digit_weights = tf.matmul(hidden_layer, hidden_weights) + hidden_biases

    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=digit_weights, labels=target_labels))
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(loss_function)

    correct_prediction = tf.equal(tf.argmax(digit_weights, 1), tf.argmax(target_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.compat.v1.global_variables_initializer().run()

    EPOCH = 20
    BATCH_SIZE = 500
    TRAIN_DATASIZE, _ = load_pre_pross.x_train.shape
    PERIOD = TRAIN_DATASIZE // BATCH_SIZE
    # this for loop iterates over the steps or epochs that the model takes to train.


class epochs_training:
    original_learning_rate = '0.5'

    for e in range(NN_model_trainer.EPOCH):
        idxs = npy.random.permutation(NN_model_trainer.TRAIN_DATASIZE)
        X_random = load_pre_pross.x_train[idxs]
        Y_random = load_pre_pross.y_train[idxs]

        for misclassified_data in range(NN_model_trainer.PERIOD):
            batch_X = X_random[misclassified_data * NN_model_trainer.BATCH_SIZE:(
                            misclassified_data + 1) * NN_model_trainer.BATCH_SIZE]
            batch_Y = Y_random[misclassified_data * NN_model_trainer.BATCH_SIZE:(
                            misclassified_data + 1) * NN_model_trainer.BATCH_SIZE]

            NN_model_trainer.optimizer.run(
                feed_dict={NN_model_trainer.input_images: batch_X, NN_model_trainer.target_labels: batch_Y})

        results = f"\nTraining epoch ________________________________{str(e + 1)}\n"

        misclassified_idx = \
            np.where(NN_model_trainer.correct_prediction.eval(
                feed_dict={NN_model_trainer.input_images: load_pre_pross.x_test,
                           NN_model_trainer.target_labels: load_pre_pross.y_test}) == False)[0]

        plt.figure(figsize=(15, 4))

        for misclassified_data in range(5):
            plt.subplot(1, 5, misclassified_data + 1)
            label = load_pre_pross.y_test[misclassified_idx[misclassified_data]].argmax(axis=0)
            image = load_pre_pross.x_test[misclassified_idx[misclassified_data]].reshape([28, 28])
            plt.title("misclassified image\n" + 'Label: {}\nPredicted: {}'.format(label, NN_model_trainer.digit_weights.eval(
                    feed_dict={NN_model_trainer.input_images:load_pre_pross.x_test
                              [misclassified_idx[misclassified_data]].reshape(1, 784)})[0].argmax()))
            plt.imshow(image, cmap=plt.get_cmap('gray_r'))
            plt.axis('off')
        plt.show()

        final_accuracy = NN_model_trainer.accuracy.eval(feed_dict={NN_model_trainer.input_images:
                         load_pre_pross.x_test, NN_model_trainer.target_labels: load_pre_pross.y_test})

        result_2 = f"Learning Rate: {original_learning_rate}\n"
        result_3 = f"Batch Size: {NN_model_trainer.BATCH_SIZE}\n"
        result_4 = "Overall Accuracy: {:.2f}".format(final_accuracy)

        print(results + result_2 + result_3 + result_4)
