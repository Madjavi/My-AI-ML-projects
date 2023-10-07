import numpy as npy
import tensorflow as tf
import matplotlib.pyplot as plt
from os import system, name


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


# This class defines the fixed seed needed for random numbers prediction.
class rand_num:
    npy.random.seed(101)
    tf.compat.v1.set_random_seed(101)


# This class generates random training data for the regression model.
class rand_num_gen:
    clr_T()
    # Generates random linear data.
    x = npy.linspace(0, 50, 50)
    y = npy.linspace(0, 50, 50)
    # Adding noise to the random linear data.
    x += npy.random.uniform(-4, 4, 50)
    y += npy.random.uniform(-4, 4, 50)

    n = len(x)  # Number of data points.

    # Plot the training data.
    plt.scatter(x, y, label='Training Data', color='red')

    plt.xlabel('Input Data')
    plt.ylabel('Target Data')
    plt.title('Training Data Plot')

    plt.legend()
    plt.show()

    tf.compat.v1.disable_eager_execution()  # This line disables the eager execution which is by default in TF v2.

    # Defines the placeholder.
    X = tf.compat.v1.placeholder(tf.float32)
    Y = tf.compat.v1.placeholder(tf.float32)

    # Declare two trainable TensorFlow variables for the weights and bias.
    weights = tf.Variable(npy.random.randn(), dtype=tf.float32)
    bias = tf.Variable(npy.random.randn(), dtype=tf.float32)

    # Hyperparameters of the model.
    learning_rate = 0.001
    training_epoch = 1000

    # hypothesis
    y_pre = tf.add(tf.multiply(X, weights), bias)

    # Cost function
    cost_f = tf.reduce_sum(tf.pow(y_pre - Y, 2)) / (2 * n)

    # Optimizer implementation
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost_f)

    # Implement the training session
    with tf.compat.v1.Session() as session:
        session.run(tf.compat.v1.global_variables_initializer())

        # Training process.
        for epochs in range(training_epoch):
            # Optimizer and cost function
            _, c = session.run([optimizer, cost_f], feed_dict=
            {X: x, Y: y})

            # Number of steps (epochs)
            if (epochs + 1) % 100 == 0:
                # Epochs count
                epochs_count = f"EPOCH {epochs + 1}/{training_epoch}==================================== Cost: {c:.4f}"
                print(epochs_count)
                # Print the cost
                training = f"Cost of training: {c:.4f}\n"
                print(training)
        # Final weights and bias.
        weights, bias = session.run([weights, bias])

        weights_1 = f"Weights: {weights:.4f}"
        bias_1 = f"\nBias: {bias:.4f}"
        # Print the cost
        print(weights_1, bias_1)


# This class plot the fitted line in top of the original.
class plotting:
    plt.scatter(rand_num_gen.x, rand_num_gen.y)
    # Fitted line in top of the data points.
    plt.plot(rand_num_gen.x, rand_num_gen.weights * rand_num_gen.x + rand_num_gen.bias,
             color='red', label='Fitted Line')
    plt.xlabel('x_train')
    plt.ylabel('y_train')
    plt.title('Fitted line on top of Original data')
    plt.legend()
    plt.show()


# executes the entire code.
if __name__ == "main":
    rand_num()
    rand_num_gen()
    plotting()
