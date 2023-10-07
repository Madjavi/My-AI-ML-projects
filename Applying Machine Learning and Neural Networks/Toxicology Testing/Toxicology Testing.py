# Imported Libraries
import numpy as npy
import tensorflow as tf
from tensorboard import program
import argparse
import matplotlib.pyplot as plt
from deepchem import deepchem as dc
from sklearn.metrics import accuracy_score


class Tox21_dataset:
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str, default='Tox21',
                        help="This is the name of the dataset.")

    arg = parser.parse_args()

    name = arg.name

    tracking_location = "/Project Py/fcnet-tox21"

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

    # Defined placeholders that accept mini-batches of different sizes.
    # Defined the fully connected architecture.
    with tf.name_scope("placeholders"):
        x = tf.compat.v1.placeholder(tf.float32, (None, d))
        y = tf.compat.v1.placeholder(tf.float32, (None,))
        keep_prob = tf.compat.v1.placeholder(tf.float32)

    # Hidden layers with added dropout.
    with tf.name_scope("hidden-layer"):
        W = tf.compat.v1.Variable(tf.compat.v1.random_normal((d, n_hidden)))
        b = tf.compat.v1.Variable(tf.compat.v1.random_normal((n_hidden,)))
        x_hidden = tf.compat.v1.nn.relu(tf.matmul(x, W) + b)
        x_hidden = tf.compat.v1.nn.dropout(x_hidden, keep_prob)

    with tf.name_scope("output"):
        W = tf.compat.v1.Variable(tf.compat.v1.random_normal((n_hidden, 1)))
        b = tf.compat.v1.Variable(tf.compat.v1.random_normal((1,)))
        y_logit = tf.compat.v1.matmul(x_hidden, W) + b
        # Provides a probability of one to the class.
        y_one_prob = tf.compat.v1.sigmoid(y_logit)
        # Rounding P(y=1) will give the correct prediction.
        y_pred = tf.compat.v1.round(y_one_prob)

    with tf.name_scope("loss"):
        # Compute the cross-entropy term for each datapoint
        y_expand = tf.compat.v1.expand_dims(y, 1)
        entropy = tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
        # Sum all contributions
        L = tf.compat.v1.reduce_sum(entropy)

    with tf.name_scope("optim"):
        train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(L)

    with tf.name_scope("summaries"):
        tf.compat.v1.summary.scalar("loss", L)
        merged = tf.compat.v1.summary.merge_all()

    train_writer = tf.compat.v1.summary.FileWriter('/Project Py/fcnet-tox21')

    N = train_X.shape[0]


# Mini-batches training implementation
class train_and_eval:
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        step = 0
        for epoch in range(Tox21_dataset.n_epoch):
            pos = 0
            while pos < Tox21_dataset.N:
                batch_X = Tox21_dataset.train_X[pos:pos + Tox21_dataset.batch_size]
                batch_y = Tox21_dataset.train_y[pos:pos + Tox21_dataset.batch_size]
                feed_dict = {Tox21_dataset.x: batch_X,
                             Tox21_dataset.y: batch_y, Tox21_dataset.keep_prob: Tox21_dataset.dropout_prob}
                _, summary, loss = sess.run([Tox21_dataset.train_op, Tox21_dataset.merged, Tox21_dataset.L],
                                            feed_dict=feed_dict)
                status = f"epoch {epoch}, step {step}, loss: {loss}"
                print(status)
                Tox21_dataset.train_writer.add_summary(summary, step)

                step += 1
                pos += Tox21_dataset.batch_size

            train_y_pred = sess.run(Tox21_dataset.y_pred,
                                    feed_dict={Tox21_dataset.x: Tox21_dataset.train_X, Tox21_dataset.keep_prob: 1.0})
            valid_y_pred = sess.run(Tox21_dataset.y_pred,
                                    feed_dict={Tox21_dataset.x: Tox21_dataset.valid_X, Tox21_dataset.keep_prob: 1.0})
            train_weighted_score = accuracy_score(Tox21_dataset.train_y,
                                                  train_y_pred, sample_weight=Tox21_dataset.train_w)
            valid_weighted_score = accuracy_score(Tox21_dataset.valid_y,
                                                  valid_y_pred, sample_weight=Tox21_dataset.valid_w)

        valid_y_pred = sess.run(Tox21_dataset.y_pred,
                                feed_dict={Tox21_dataset.x: Tox21_dataset.valid_X, Tox21_dataset.keep_prob: 1.0})
        valid_accuracy = accuracy_score(Tox21_dataset.valid_y, valid_y_pred)
        test_y_pred = sess.run(Tox21_dataset.y_pred, feed_dict={Tox21_dataset.x: Tox21_dataset.test_X,
                                                                Tox21_dataset.keep_prob: 1.0})
        test_weighted_score = accuracy_score(Tox21_dataset.test_y, test_y_pred, sample_weight=Tox21_dataset.test_w)

    Tox21_dataset.train_writer.close()


class outputs:
    # Prediction outputs.
    result1 = f"\nTraining Model: {Tox21_dataset.name}\n"
    result2 = f"Train Weighted Classification Accuracy: {train_and_eval.train_weighted_score}\n"
    result3 = f"Valid Weighted Classification Accuracy: {train_and_eval.valid_weighted_score}\n"
    result4 = f"Test Weighted Classification Accuracy: {train_and_eval.test_weighted_score}"
    print(result1 + result2 + result3 + result4)


# This code runs the entire code plus provides the tensorboard command and url.
if __name__ == "__main__":
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', Tox21_dataset.tracking_location])
    url = tb.launch()
    note = "\nRun TensorBoard by entering the following command in the terminal or " \
           "windows command prompt: tensorboard --logdir=/tmp/fcnet-tox21"
    message = f"\nTensorBoard listening on url: {url}"
    print(note + message)
