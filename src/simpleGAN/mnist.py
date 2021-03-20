import argparse
import tensorflow as tf
import numpy as np
import os
import cv2
from sklearn.utils import shuffle
from imutils import build_montages

from Discriminator import Discriminator
from Generator import Generator


parser = argparse.ArgumentParser(description='Simple GAN for mnist')
parser.add_argument('--z-dim', type=int, help="Path to the environment configuration", default=10)
parser.add_argument('--hidden-dim', type=int, help="Path to checkpoint", default=128)
parser.add_argument('--lr', type=float, help="Path to checkpoint", default=0.00001)
parser.add_argument('--epochs', type=int, help="Path to checkpoint", default=200)
parser.add_argument('--batch-size', type=int, help="Path to checkpoint", default=128)
parser.add_argument('--output', type=str, help="Path to output", default="../../output")
args = parser.parse_args()


# Data preparation
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], -1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

im_dim = train_images.shape[-1]

# Build the generator
gen = Generator(im_dim=im_dim, hidden_dim=args.hidden_dim).gen

# Build the discriminator
disc = Discriminator(hidden_dim=args.hidden_dim).disc
disc_opt = tf.keras.optimizers.Adam(lr=args.lr)

# Compile the discriminative network
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
disc.compile(loss=cross_entropy, optimizer=disc_opt)

# Build the adversarial model by first setting the discriminator
disc.trainable = False
ganInput = tf.keras.Input(shape=(args.z_dim,))
ganOutput = disc(gen(ganInput))
gan = tf.keras.Model(ganInput, ganOutput)

# Compile the GAN
gan_opt = tf.keras.optimizers.Adam(lr=args.lr)
gan.compile(loss=cross_entropy, optimizer=gan_opt)

# Randomly generate some benchmark noise so we can consistently visualize how the generative modeling is learning
benchmarkNoise = np.random.uniform(-1, 1, size=(args.batch_size, args.z_dim))

for epoch in range(0, args.epochs):
    # show epoch information and compute the number of batches per
    # epoch
    print("[INFO] starting epoch {} of {}...".format(epoch + 1, args.epochs))
    batchesPerEpoch = int(train_images.shape[0] / args.batch_size)

    for i in range(0, batchesPerEpoch):
        # initialize an (empty) output path
        p = None

        # select the next batch of images, then randomly generate
        # noise for the generator to predict on
        imageBatch = train_images[i * args.batch_size:(i + 1) * args.batch_size]
        noise = np.random.uniform(-1, 1, size=(args.batch_size, args.z_dim))

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
        disc_loss = disc.train_on_batch(X, y)

        # let's now train our generator via the adversarial model by
        # (1) generating random noise and (2) training the generator
        # with the discriminator weights frozen
        noise = np.random.uniform(-1, 1, (args.batch_size, args.z_dim))
        fakeLabels = [1] * args.batch_size
        fakeLabels = np.reshape(fakeLabels, (-1,))
        gan_loss = gan.train_on_batch(noise, fakeLabels)

        # check to see if this is the end of an epoch, and if so,
        # initialize the output path
        if i == batchesPerEpoch - 1:
            p = [args.output, "epoch_{}_output.png".format(str(epoch + 1).zfill(4))]

        # otherwise, check to see if we should visualize the current
        # batch for the epoch
        else:
            # visualizations later in the training process are less
            # interesting
            if i % 500 == 0:
                p = [args.output, "epoch_{}_step_{}.png".format(str(epoch + 1).zfill(4), str(i).zfill(5))]

        # check to see if we should visualize the output of the
        # generator model on our benchmark data
        if p is not None:
            # show loss information
            print("[INFO] Step {}_{}: discriminator_loss={:.6f}, adversarial_loss={:.6f}".format(
                epoch + 1, i, disc_loss, gan_loss))

            # make predictions on the benchmark noise, scale it back
            # to the range [0, 255], and generate the montage
            images = gen.predict(benchmarkNoise)
            images = ((images * 127.5) + 127.5).astype("uint8")
            images = images.reshape(args.batch_size, 28, 28, 1)
            images = np.repeat(images, 3, axis=-1)
            vis = build_montages(images, (28, 28), (16, 16))[0]

            # write the visualization to disk
            p = os.path.sep.join(p)
            cv2.imwrite(p, vis)








