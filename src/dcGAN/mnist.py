import argparse
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import os
import cv2
from sklearn.utils import shuffle
from imutils import build_montages
from matplotlib import pyplot as plt

from dcGAN.Discriminator import Discriminator
from dcGAN.Generator import Generator


parser = argparse.ArgumentParser(description='Simple GAN for mnist')
parser.add_argument('--z-dim', type=int, help="Noise dimension", default=64)
parser.add_argument('--lr', type=float, help="Learning rate", default=0.0002)
parser.add_argument('--epochs', type=int, help="Epochs", default=2)
parser.add_argument('--batch-size', type=int, help="Batch size", default=128)
parser.add_argument('--output', type=str, help="Path to output", default="../../output/dcGAN")
args = parser.parse_args()


def plot_history(d_hist, g_hist, a_hist, folder="../../output/"):
    """
    Plot history of training
    """
    plt.subplot(2, 1, 1)
    plt.plot(d_hist, label='dis')
    plt.plot(g_hist, label='gen')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(a_hist, label='acc')
    plt.legend()

    plt.savefig(os.path.join(folder, "plot_line_plot_loss.png"))
    plt.close()


# Data preparation
(trainX, _), (testX, _) = tf.keras.datasets.mnist.load_data()

train_images = np.concatenate([trainX, testX], axis=0)

train_images = train_images.reshape((len(train_images), 28, 28, 1))
train_images = (train_images.astype("float") - 127.5) / 127.5

# Build the generator
generator = Generator(im_chan=1, hidden_dim=64)
gen = generator.gen

# Build the discriminator
disc = Discriminator(hidden_dim=16).disc
disc_opt = tf.keras.optimizers.Adam(lr=args.lr, beta_1=0.5, beta_2=0.999)

# Compile the discriminative network
disc.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=disc_opt, metrics=['accuracy'])

# Build the adversarial model by first setting the discriminator
ganInput = tf.keras.Input(shape=(1, 1, args.z_dim))
disc.trainable = False
ganOutput = disc(gen(ganInput))
gan = tf.keras.Model(ganInput, ganOutput)

# Compile the GAN
gan_opt = tf.keras.optimizers.Adam(lr=args.lr, beta_1=0.5, beta_2=0.999)
gan.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer=gan_opt)

d_hist, g_hist, a_hist = list(), list(), list()

for epoch in range(0, args.epochs):
    # show epoch information and compute the number of batches per
    # epoch
    print("[INFO] starting epoch {} of {}...".format(epoch + 1, args.epochs))
    batchesPerEpoch = int(train_images.shape[0] / args.batch_size)

    for i in tqdm(range(0, batchesPerEpoch)):
        # select the next batch of images, then randomly generate
        # noise for the generator to predict on
        imageBatch = train_images[i * args.batch_size:(i + 1) * args.batch_size]
        noise = generator.get_noise(args.batch_size, args.z_dim)

        # generate images using the noise + generator model
        genImages = gen.predict(noise, verbose=0)

        # concatenate the *actual* images and the *generated* images,
        # construct class labels for the discriminator, and shuffle
        # the data
        X = np.concatenate((imageBatch, genImages))
        y = ([1] * args.batch_size) + ([0] * args.batch_size)
        y = np.reshape(y, (-1,))
        (X, y) = shuffle(X, y)

        # train the discriminator on the data
        disc_loss, disc_acc = disc.train_on_batch(X, y)

        # let's now train our generator via the adversarial model by
        # (1) generating random noise and (2) training the generator
        # with the discriminator weights frozen
        noise = generator.get_noise(args.batch_size, args.z_dim)
        fakeLabels = [1] * args.batch_size
        fakeLabels = np.reshape(fakeLabels, (-1,))
        gan_loss = gan.train_on_batch(noise, fakeLabels)

        d_hist.append(disc_loss)
        g_hist.append(gan_loss)
        a_hist.append(disc_acc)

        # check to see if this is the end of an epoch
        if i == batchesPerEpoch - 1:

            # Show loss information
            print("""
                [INFO] Step {}_{}: discriminator_loss={:.6f}, adversarial_loss={:.6f}, discriminator_accuracy={}"
            """.format(epoch + 1, i, disc_loss, gan_loss, int(disc_acc * 100)))

            # Make predictions on the benchmark noise
            noise = generator.get_noise(256, args.z_dim)
            images_fake = gen.predict(noise)
            images_fake = ((images_fake * 127.5) + 127.5).astype("uint8")
            images_fake = images_fake.reshape(256, 28, 28, 1)
            images_fake = np.repeat(images_fake, 3, axis=-1)

            vis_fake = build_montages(images_fake, (28, 28), (16, 16))

            # Write the visualizations to disk
            p_fake = os.path.sep.join([args.output, "epoch_{}_fake.png".format(str(epoch + 1).zfill(4))])
            cv2.imwrite(p_fake, vis_fake[0])

plot_history(d_hist, g_hist, a_hist, folder="../../output/dcGAN")