# Imported libraries
import random
from numpy import zeros
import numpy as npy
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import time as tm
import timeit as ti
from keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout

"This implementation is based on a Generative Adversarial Network (GAN) architecture develop by "
"Brownlee J. (2019). How to Develop a GAN to Generate CIFAR10 Small Color Photographs."


# load and prepare cifar10 training images
def load_dataset():
    ver = tf.__version__
    msg_0_0 = f"\nWorking with a Generative Adversarial Network Using TensorFlow and Keras version: {ver}\n"
    print(msg_0_0)
    tm.sleep(1)
    # Progress bar.
    for progress in tqdm(range(101), desc="Loading..", ascii=False, ncols=75):
        tm.sleep(0.01)

    print("\nStarting----->\n")
    tm.sleep(1)

    # load cifar10 dataset
    (X_train, y), (_, _) = tf.keras.datasets.cifar10.load_data()
    # convert from unsigned ints to floats
    X = X_train.astype('float32')
    # scale from [0,255] to [-1,1]
    X = (X - 127.5) / 127.5

    msg_0 = X_train.shape, y.shape, _.shape, _.shape

    # Dataset progress bar.
    for progress_2 in tqdm(range(101), desc="Loading Dataset", ascii=False, ncols=75):
        tm.sleep(0.01)

    msg_1 = f"\nCIFAR10 dataset--------------------------------------->>\n{msg_0}\n\n"
    print(msg_1)
    tm.sleep(1)

    return X


# The function below defines the discriminator neural network using keras API through tensorflow.
def discriminator(input_shape=(32, 32, 3)):
    discriminator_net = tf.keras.Sequential()
    discriminator_net.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    discriminator_net.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    discriminator_net.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    discriminator_net.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    discriminator_net.add(tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    discriminator_net.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    discriminator_net.add(tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    discriminator_net.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    discriminator_net.add(tf.keras.layers.Flatten())
    discriminator_net.add(tf.keras.layers.Dropout(0.4))
    discriminator_net.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    optimization = Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator_net.compile(loss='binary_crossentropy', optimizer=optimization, metrics=['Accuracy'])
    return discriminator_net


# The function below defines the generator neural network using keras API through tensorflow.
def generator(latent_dim):
    generator_net = tf.keras.Sequential()
    neurons = 256 * 4 * 4
    generator_net.add(tf.keras.layers.Dense(neurons, input_dim=latent_dim))
    generator_net.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    generator_net.add(tf.keras.layers.Reshape((4, 4, 256)))
    generator_net.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    generator_net.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    generator_net.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    generator_net.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    generator_net.add(tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    generator_net.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    generator_net.add(tf.keras.layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
    return generator_net


# this function combines both networks and creates the GAN that shall update the generator.
def GAN(gen_net, dis_net):
    dis_net.trainable = False
    GANs = tf.keras.Sequential()
    GANs.add(gen_net)
    GANs.add(dis_net)
    optimizer = Adam(learning_rate=0.0002, beta_1=0.1)
    GANs.compile(loss='binary_crossentropy', optimizer=optimizer)
    return GANs


# this function choose random instances to select the real images.
def real_samples_gen(dataset, images):
    ran = randint(0, dataset.shape[0], images)
    ran_imgs = dataset[ran]
    smpl = npy.ones((images, 1))
    return ran_imgs, smpl


# this function send the input data to the generator nn.
def input_points(latent_dims, images):
    x_in = randn(latent_dims * images)
    x_in = x_in.reshape(images, latent_dims)
    return x_in


# this function generates fake samples that are injected into the discriminator nn.
def fake_sample(gen_net, latent_dims, images):
    x_in = input_points(latent_dims, images)
    X_img = gen_net.predict(x_in)
    y_img = zeros((images, 1))
    return X_img, y_img


# create and save a plot of generated images
def plotting_img(samples, epoch, num=7):
    samples = (samples + 1) / 2.0

    for data in range(num * num):
        plt.subplot(num, num, 1 + data)
        plt.axis('off')
        plt.imshow(samples[data])

    new_file_name = 'generated_images_e%03d.png' % (epoch + 1)
    plt.savefig(new_file_name)
    plt.close()


# this function displays the generated images.
# every 40 epochs.
def image_plotter():
    row, col = 4, 4
    noise = npy.random.normal(0, 1, (row * col, latent_dims))
    generated_image = gen_net.predict(noise)
    generated_image = 0.5 * generated_image + 0.5
    fig, axs = plt.subplots(row, col)
    count = 0

    for i in range(row):
        for j in range(col):
            axs[i, j].imshow(generated_image[count])
            axs[i, j].axis('off')
            count += 1

    plt.show()
    plt.close()


# This function evaluated the discriminator.
def eval_models(epoch, gen_net, dis_net, dataset, latent_dims, images=150):
    X_real, Y_real = real_samples_gen(dataset, images)
    _, eval_real = dis_net.evaluate(X_real, Y_real, verbose=0)
    x_fake, y_fake = fake_sample(gen_net, latent_dims, images)
    _, eval_fake = dis_net.evaluate(x_fake, y_fake, verbose=0)
    accuracy = 'Accuracy real: %.0f%%, fake: %.0f%%' % (eval_real * 100, eval_fake * 100)
    print(accuracy)
    plotting_img(x_fake, epoch)
    filename = 'generator_model_%03d.h5' % (epoch + 1)
    gen_net.save(filename)


# this function trains the discriminator and the generator networks.
def training(gen_net, dis_net, GAN, dataset, latent_dims, num_epochs=200, num_batch=128):
    batch_per_epoch = int(dataset.shape[0] / num_batch)
    min_batch = int(num_batch / 2)
    display_interval = 20

    for epoch in range(num_epochs):

        if epoch % display_interval == 0:
            image_plotter()

        for batch in range(batch_per_epoch):
            X_real, y_real = real_samples_gen(dataset, min_batch)
            dis_loss_1 = dis_net.train_on_batch(X_real, y_real)

            x_fake, y_fake = fake_sample(gen_net, latent_dims, min_batch)
            dis_loss_2 = dis_net.train_on_batch(x_fake, y_fake)

            x_GAN = input_points(latent_dims, num_batch)
            y_GAN = npy.ones((num_batch, 1))

            gen_loss = GAN.train_on_batch(x_GAN, y_GAN)

            tqdm.write(f"\nEpoch: {epoch} | \nGAN Loss: {gen_loss} "
                       f"| \nReal Loss: {dis_loss_1} | \nFake Loss: {dis_loss_2}\n")

        if (epoch + 1) % 10 == 0:
            eval_models(epoch, gen_net, dis_net, dataset, latent_dims)

    last = npy.random.normal(size=(40, latent_dims))
    generated_images = gen_net.predict(last)
    generated_images = 0.5 * generated_images + 0.5

    f, ax = plt.subplots(5, 8, figsize=(16, 10))
    for img_data_2, image in enumerate(generated_images):
        ax[img_data_2 // 8, img_data_2 % 8].imshow(image)
        ax[img_data_2 // 8, img_data_2 % 8].axis('off')

    plt.show()
    plt.savefig(f"Image generates from the last epoch.jpg")
    plt.close()


# Executes the Algorithm.
if __name__ == "__main__":
    latent_dims = 100
    gen_net = generator(latent_dims)
    dis_net = discriminator()
    GAN = GAN(gen_net, dis_net)
    dataset = load_dataset()
    training(gen_net, dis_net, GAN, dataset, latent_dims)
