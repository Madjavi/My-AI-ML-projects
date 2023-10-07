# Imported Libraries
import time as tm
import numpy as npy
import tensorflow as tf
from tensorboard import program
import argparse
import matplotlib.pyplot as plt
from os import system, name
from deepchem import deepchem as dc
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# This class houses the messages display on the console when interacting with the algorithm.
class info_data:
    info = {'Option1': "\nHidden: ",
            'Option2': "Layers: ",
            'Option3': "Learning rate: ",
            'Option4': "Dropout Prob: ",
            'Option5': "Epochs: ",
            'Option6': "Do you want to proceed? (Enter 'quit' to exit): ",
            'Instruction1': "\nEnter the new hyperparameters for the TensorFlow model "
                            "to benchmark the model's best performance against the random forest classifier.\n"}


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


# This class loads the dataset and runs the baseline utilized to compare the performance accuracy of the model against
# the different values that will be introduced in a nested loop in order to track the best performer.
class Tox21_dataset:
    # The argument parser provides a name argument to the console.
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default='Improving the Accuracy of Tox21 Neural Network',
                        help="This is the name of the dataset.")

    arg = parser.parse_args()

    name = arg.name

    tracking_location = "/Project Py/DATA/fcnet-tox21"

    npy.random.seed(456)
    tf.compat.v1.set_random_seed(456)

    # Loads the Toxi21 dataset.
    _, (train, valid, test), _ = dc.molnet.load_tox21()
    train_X, train_y, train_w = train.X, train.y, train.w
    valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
    test_X, test_y, test_w = test.X, test.y, test.w

    # Remove Extra tasks.
    train_y = train_y[:, 0]
    valid_y = valid_y[:, 0]
    test_y = test_y[:, 0]
    train_w = train_w[:, 0]
    valid_w = valid_w[:, 0]
    test_w = test_w[:, 0]

    # Generates TensorFLow Graphs.
    d = 1024
    n_hidden = 50
    learning_rate = 0.001
    n_epoch = 10
    batch_size = 100
    dropout_prob = 0.5

    tf.compat.v1.disable_eager_execution()

    # Generate a graph using a random forest classifier.
    sklearn_model = RandomForestClassifier(class_weight="balanced", n_estimators=50)
    sklearn_msg = "\nRandom Forest Classifier model baseline.\n"
    sklearn_msg_2 = "\n*******Tox21 Random Forest Classifier model Accuracy performance results*******\n"

    sklearn_model.fit(train_X, train_y)

    train_y_pred = sklearn_model.predict(train_X)
    valid_y_pred = sklearn_model.predict(valid_X)
    test_y_pred = sklearn_model.predict(test_X)

    print(sklearn_msg + sklearn_msg_2)

    weighted_score_1 = accuracy_score(train_y, train_y_pred, sample_weight=train_w)
    weighted_score_2 = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
    weighted_score_3 = accuracy_score(test_y, test_y_pred, sample_weight=test_w)

    msg_0 = f"\nRandom Forest Classifier Hyperparameters: n_hidden = {n_hidden}, " \
            f"learning_rate = {learning_rate}, n_epochs = {n_epoch}, " \
            f"batches_size = {batch_size}, dropout_prob = {dropout_prob}\n\n"

    msg_1 = f"Weighted train Classification Accuracy: {weighted_score_1}"
    msg_2 = f"\nWeighted valid Classification Accuracy: {weighted_score_2}"
    msg_3 = f"\nWeighted test Classification Accuracy: {weighted_score_3}\n"
    msg_4 = "\n-----------------------------------------------\n"

    print(msg_0 + msg_1 + msg_2 + msg_3 + msg_4)

    tm.sleep(3)


# The function shall evaluate the performance of the model with different hyperparameter.
def eval_tox21_hyperparams(n_hidden=50, n_layers=1, learning_rate=.001,
                           dropout_prob=0.5, n_epochs=45, batch_size=100,
                           weight_positives=True):
    hype1 = "\n---------------------------------------------"
    hype2 = "\nModel hyperparameters"
    hype3 = f"\nn_hidden = {n_hidden}"
    hype4 = f"\nn_layers = {n_layers}"
    hype5 = f"\nlearning_rate = {learning_rate}"
    hype6 = f"\nn_epochs = {n_epochs}"
    hype7 = f"\nbatch_size = {batch_size}"
    hype8 = f"\nweight_positives = {str(weight_positives)}"
    hype9 = f"\ndropout_prob = {dropout_prob}"
    hype10 = "\n---------------------------------------------"
    print(hype1 + hype2 + hype3 + hype4 + hype5 + hype6 + hype7 + hype8 + hype9 + hype10)

    d = 1024
    graph = tf.compat.v1.Graph()
    with graph.as_default():
        _, (train, valid, test), _ = dc.molnet.load_tox21()
        train_X, train_y, train_w = train.X, train.y, train.w
        valid_X, valid_y, valid_w = valid.X, valid.y, valid.w
        test_X, test_y, test_w = test.X, test.y, test.w

        # Remove extra tasks
        train_y = train_y[:, 0]
        valid_y = valid_y[:, 0]
        test_y = test_y[:, 0]
        train_w = train_w[:, 0]
        valid_w = valid_w[:, 0]
        test_w = test_w[:, 0]

        # Generate tensorflow graph
        with tf.compat.v1.name_scope("placeholders"):
            x = tf.compat.v1.placeholder(tf.float32, (None, d))
            y = tf.compat.v1.placeholder(tf.float32, (None,))
            w = tf.compat.v1.placeholder(tf.float32, (None,))
            keep_prob = tf.compat.v1.placeholder(tf.float32)
        for layer in range(n_layers):
            with tf.compat.v1.name_scope("layer-%d" % layer):
                W = tf.compat.v1.Variable(tf.compat.v1.random_normal((d, n_hidden)))
                b = tf.compat.v1.Variable(tf.compat.v1.random_normal((n_hidden,)))
                x_hidden = tf.compat.v1.nn.relu(tf.matmul(x, W) + b)
                # Apply dropout
                x_hidden = tf.compat.v1.nn.dropout(x_hidden, keep_prob)
        with tf.compat.v1.name_scope("output"):
            W = tf.compat.v1.Variable(tf.compat.v1.random_normal((n_hidden, 1)))
            b = tf.compat.v1.Variable(tf.compat.v1.random_normal((1,)))
            y_logit = tf.compat.v1.matmul(x_hidden, W) + b
            # the sigmoid gives the class probability of 1
            y_one_prob = tf.compat.v1.sigmoid(y_logit)
            # Rounding P(y=1) will give the correct prediction.
            y_pred = tf.compat.v1.round(y_one_prob)
        with tf.compat.v1.name_scope("loss"):
            # Compute the cross-entropy term for each datapoint
            y_expand = tf.compat.v1.expand_dims(y, 1)
            entropy = tf.compat.v1.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
            # Multiply by weights
            if weight_positives:
                w_expand = tf.compat.v1.expand_dims(w, 1)
                entropy = w_expand * entropy
            # Sum all contributions
            L = tf.compat.v1.reduce_sum(entropy)

        with tf.compat.v1.name_scope("optim"):
            train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(L)

        with tf.compat.v1.name_scope("summaries"):
            tf.compat.v1.summary.scalar("loss", L)
            merged = tf.compat.v1.summary.merge_all()

        hyper_str = "d-%d-hidden-%d-lr-%f-n_epochs-%d-batch_size-%d-weight_pos-%s" % (
            d, n_hidden, learning_rate, n_epochs, batch_size, str(weight_positives))
        train_writer = tf.compat.v1.summary.FileWriter('/Project Py/DATA/fcnet-func-' + hyper_str)
        N = train_X.shape[0]

        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())

            step = 0
            for epoch in range(Tox21_dataset.n_epoch):
                pos = 0
                while pos < N:
                    batch_X = train_X[pos:pos + batch_size]
                    batch_y = train_y[pos:pos + batch_size]
                    batch_w = train_w[pos:pos + batch_size]
                    feed_dict = {x: batch_X, y: batch_y, w: batch_w, keep_prob: dropout_prob}
                    _, summary, loss = sess.run([train_op, merged, L], feed_dict=feed_dict)
                    status = f"epoch {epoch}, step {step}, loss: {loss}"
                    print(status)
                    train_writer.add_summary(summary, step)

                    step += 1
                    pos += batch_size

                train_y_pred = sess.run(y_pred, feed_dict={x: train_X, keep_prob: 1.0})
                train_weighted_score = accuracy_score(train_y, train_y_pred, sample_weight=train_w)

                valid_y_pred = sess.run(y_pred, feed_dict={x: valid_X, keep_prob: 1.0})
                valid_accuracy = accuracy_score(valid_y, valid_y_pred)
                test_y_pred = sess.run(y_pred, feed_dict={x: test_X, keep_prob: 1.0})
                test_weighted_score = accuracy_score(test_y, test_y_pred, sample_weight=test_w)
                weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)

                # Prediction outputs.
                result1 = f"\nTraining Model: {Tox21_dataset.name}"
                result2 = f"\nTrain Weighted Classification Accuracy: {train_weighted_score}"
                result3 = f"\nTest Weighted Classification Accuracy: {test_weighted_score}"
                result4 = f"\nValid Weighted Classification Accuracy: {weighted_score}"
                result5 = f"\nModel Accuracy: {valid_accuracy}"
                print(result1 + result2 + result3 + result4 + result5)

            return valid_y_pred, test_weighted_score, valid_accuracy, test_y_pred, weighted_score


# This class shall iterate throughout the code to inject different hyperparameter values that shall serve to compare
# tha different results in order to compare the best performer.
class N_hyperparameters_T:
    msg_1 = info_data.info['Instruction1']
    print(msg_1)

    while True:
        selection_1 = int(input(info_data.info['Option1']))
        selection_2 = int(input(info_data.info['Option2']))
        selection_3 = float(input(info_data.info['Option3']))
        selection_4 = float(input(info_data.info['Option4']))
        selection_5 = float(input(info_data.info['Option5']))

        new_n_hidden = [selection_1]
        new_n_layers_ = [selection_2]
        new_learning_rates = [selection_3]
        new_dropout_prob = [selection_4]
        n_epochs = [selection_5]
        batch_size = 100
        variable_weight_positive = True, False
        repeats = 5

        Best_accuracy_S = 0.0
        Best_results = {}

        for n_hidden in new_n_hidden:
            for n_layers in new_n_layers_:
                for learning_rate in new_learning_rates:
                    for dropout in new_dropout_prob:
                        for epochs in n_epochs:
                            for weights in variable_weight_positive:
                                average_accuracy = 0.0

                                for _ in range(repeats):
                                    accuracy = eval_tox21_hyperparams(n_hidden=n_hidden, n_layers=n_layers,
                                                                      learning_rate=learning_rate,
                                                                      dropout_prob=dropout,
                                                                      n_epochs=epochs, batch_size=batch_size,
                                                                      weight_positives=weights
                                                                      )[0]

                                    average_accuracy += accuracy

                                average_accuracy /= repeats

                                if npy.all(average_accuracy >= Best_accuracy_S):
                                    Best_accuracy_S = average_accuracy
                                    Best_results = {
                                        'n_hidden': n_hidden,
                                        'n_layers': n_layers,
                                        'learning_rate': learning_rate,
                                        'dropout_prob': dropout,
                                        'weight_positives': weights}

        result_1 = f"\nTox21 Model hyperparameter tuning settings: {Best_results}"
        result_2 = f"\nTox21 Model averaging the accuracy: {Best_accuracy_S}\n"

        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', Tox21_dataset.tracking_location])
        url = tb.launch()
        note = "\nRun TensorBoard by entering the following command in the terminal or " \
               "windows command prompt: tensorboard --logdir=/tmp/fcnet-tox21"
        message = f"\nTensorBoard listening on url: {url}"

        print(result_1 + result_2 + note + message)


# This code runs the entire code plus provides the tensorboard command and url.
if __name__ == "__main__":
    clr_T()
    info_data()
