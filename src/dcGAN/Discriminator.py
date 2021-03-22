import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Activation, Reshape


class Discriminator:

    def __init__(self, hidden_dim=16):
        """
        Initialize a discriminator

        Parameters:
            hidden_dim: the dimension of hidden layers
        """
        self.disc = tf.keras.Sequential(layers=[
            self.get_discriminator_block(hidden_dim, kernel_size=4, strides=2),
            self.get_discriminator_block(hidden_dim * 2, kernel_size=4, strides=2),
            self.get_discriminator_block(1, kernel_size=4, strides=2, final_layer=True),
        ])

    @staticmethod
    def get_discriminator_block(output_channels, kernel_size=4, strides=2, final_layer=False):
        """
        Function for returning a block of the discriminator's neural network

        Parameters:
            output_channels: the dimension of the output channels
            kernel_size: the size of the kernel
            strides: the strides
            final_layer: whether the block is the last layer

        Returns:
            a discriminator neural network layer
        """
        if not final_layer:
            return tf.keras.Sequential(layers=[
                Conv2D(output_channels, kernel_size, strides=strides, padding="valid"),
                BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True),
                LeakyReLU(alpha=0.2)
            ])
        else:
            return tf.keras.Sequential(layers=[
                Conv2D(output_channels, kernel_size, strides=strides, padding="valid"),
                Activation("sigmoid"),
                Reshape((1,))
            ])