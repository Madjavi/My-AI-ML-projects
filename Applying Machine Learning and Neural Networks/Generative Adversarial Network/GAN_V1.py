# Imported libraries
import numpy as npy
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import time as tm
import timeit as ti
from keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD


# This class loads the CIFAR10 dataset to feed it to the GAN.
class dataset_load:
    ver = tf.__version__
    msg_0_0 = f"\nWorking with a Generative Adversarial Network Using TensorFlow and Keras version: {ver}\n"
    print(msg_0_0)
    tm.sleep(1)
    # Progress bar.
    for progress in tqdm(range(101), desc="Loading..", ascii=False, ncols=75):
        tm.sleep(0.01)

    print("\nStarting----->\n")
    tm.sleep(1)

    # Loads the CIFAR10 dataset
    (X, y), (_, _) = tf.keras.datasets.cifar10.load_data()

    X = X[y.flatten() == 8]

    # Loads, reads, and split the dataset.
    msg_0 = X.shape, y.shape, _.shape, _.shape

    # Input shape.
    image_shape = (32, 32, 3)
    latent_dims = 100

    # Dataset progress bar.
    for progress_2 in tqdm(range(101), desc="Loading Dataset", ascii=False, ncols=75):
        tm.sleep(0.01)

    msg_1 = f"\nCIFAR10 dataset--------------------------------------->>\n{msg_0}\n\n"
    print(msg_1)
    tm.sleep(1)


# This class houses the generator conv network.
class build_generator:
    generator_net = tf.keras.Sequential()

    generator_net.add(tf.keras.layers.Dense(128 * 8 * 8, activation='relu', input_dim=dataset_load.latent_dims))
    generator_net.add(tf.keras.layers.Reshape((8, 8, 128)))
    generator_net.add(tf.keras.layers.UpSampling2D())

    generator_net.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding="same"))
    generator_net.add(tf.keras.layers.BatchNormalization(momentum=0.78))
    generator_net.add(tf.keras.layers.Activation("relu"))
    generator_net.add(tf.keras.layers.UpSampling2D())

    generator_net.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding="same"))
    generator_net.add(tf.keras.layers.BatchNormalization(momentum=0.78))
    generator_net.add(tf.keras.layers.Activation("relu"))

    generator_net.add(tf.keras.layers.Conv2D(3, kernel_size=3, padding="same"))
    generator_net.add(tf.keras.layers.Activation("tanh"))

    noise = tf.keras.Input(shape=(dataset_load.latent_dims,))
    image = generator_net(noise)

    generator_net.summary()

    sent1 = Model(noise, image)


# This class houses the discriminator network.
class build_discriminator:
    discriminator_net = tf.keras.Sequential()

    discriminator_net.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, input_shape=dataset_load.image_shape,
                                                 padding="same"))
    discriminator_net.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    discriminator_net.add(tf.keras.layers.Dropout(0.25))

    discriminator_net.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    discriminator_net.add(tf.keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1))))
    discriminator_net.add(tf.keras.layers.BatchNormalization(momentum=0.82))
    discriminator_net.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    discriminator_net.add(tf.keras.layers.Dropout(0.25))

    discriminator_net.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    discriminator_net.add(tf.keras.layers.BatchNormalization(momentum=0.82))
    discriminator_net.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    discriminator_net.add(tf.keras.layers.Dropout(0.25))

    discriminator_net.add(tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
    discriminator_net.add(tf.keras.layers.BatchNormalization(momentum=0.8))
    discriminator_net.add(tf.keras.layers.LeakyReLU(alpha=0.25))
    discriminator_net.add(tf.keras.layers.Dropout(0.25))

    discriminator_net.add(tf.keras.layers.Flatten())
    discriminator_net.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    image = tf.keras.Input(shape=dataset_load.image_shape)
    validity = discriminator_net(image)

    discriminator_net.summary()

    sent2 = Model(image, validity)


# this function displays the generated images.
# every 2500 epochs.
def image_plotter():
    r, c = 4, 4

    noise = npy.random.normal(0, 1, (r * c, dataset_load.latent_dims))
    generated_image = GAN.generator.predict(noise)

    generated_image = 0.5 * generated_image + 0.5

    fig, axs = plt.subplots(r, c)
    count = 0

    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(generated_image[count])
            axs[i, j].axis('off')
            count += 1

    plt.show()
    plt.close()


# This class builds the GAN by combining the networks.
class GAN:
    discriminator = build_discriminator.discriminator_net

    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5),
                          metrics=['Accuracy'])

    build_discriminator.discriminator_net.trainable = False

    generator = build_generator.generator_net
    z = tf.keras.Input(shape=(dataset_load.latent_dims,))
    image = generator(z)

    validity = build_discriminator.discriminator_net(image)

    combined_network = Model(z, validity)
    combined_network.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))


# this class starts the training process on the generator and the discriminator.
class Net_training:
    num_epochs = 15000
    batch_size = 32
    display_interval = 2500
    losses = []

    X = (dataset_load.X / 127.5) - 1

    valid = npy.ones((batch_size, 1))

    valid += 0.05 * npy.random.random(valid.shape)

    fake = npy.zeros((batch_size, 1))
    fake += 0.05 * npy.random.random(fake.shape)

    for epoch in range(num_epochs):
        index = npy.random.randint(0, X.shape[0], batch_size)
        images = X[index]

        noise = npy.random.normal(0, 1, (batch_size, dataset_load.latent_dims))

        generated_images = GAN.generator.predict(noise)

        discm_loss_real = GAN.discriminator.train_on_batch(images, valid)
        discm_loss_fake = GAN.discriminator.train_on_batch(generated_images, fake)
        discm_loss = 0.5 * npy.add(discm_loss_real, discm_loss_fake)

        genr_loss = GAN.combined_network.train_on_batch(noise, valid)

        if epoch % display_interval == 0:
            image_plotter()

        # displays the respective number of losses per epochs.
        tqdm.write(f"\nEpoch: {epoch} | \nGAN Loss: {genr_loss} "
                   f"| \nReal Loss: {discm_loss_real} | \nFake Loss: {discm_loss_fake}\n")

    # plots the images from the last epoch.
    noise = npy.random.normal(size=(40, dataset_load.latent_dims))

    generated_images = GAN.generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5

    f, ax = plt.subplots(5, 8, figsize=(16, 10))
    for img_data_2, image in enumerate(generated_images):
        ax[img_data_2 // 8, img_data_2 % 8].imshow(image)
        ax[img_data_2 // 8, img_data_2 % 8].axis('off')

    plt.show()
    plt.savefig(f"Image generates from the last epoch.jpg")
    plt.close()


# runtime evaluation
class model_eval:
    runtime_duration = ti.timeit(build_generator, number=1)
    runtime_duration_2 = ti.timeit(build_discriminator, number=2)
    GAN_Runtime = {'Generator_runtime': f"\nGenerator: {runtime_duration}",
                   'Discriminator_runtime': f"\nDiscriminator: {runtime_duration_2}"}

    print(GAN_Runtime['Generator_runtime'] + GAN_Runtime['Discriminator_runtime'])


if __name__ == "__main__":
    dataset_load()
    model_eval()
