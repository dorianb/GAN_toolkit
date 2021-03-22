import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Activation, Lambda


class Generator:

    def __init__(self, im_chan=1, hidden_dim=64):
        """
        Initialize a generator

        Parameters:
             im_chan: the image channel dimension
             hidden_dim: the hidden layers dimension
        """
        self.gen = tf.keras.Sequential(layers=[
            # Lambda(self._print),
            self.get_generator_block(hidden_dim * 4, kernel_size=3, strides=2),
            # Lambda(self._print),
            self.get_generator_block(hidden_dim * 2, kernel_size=4, strides=1),
            # Lambda(self._print),
            self.get_generator_block(hidden_dim, kernel_size=3, strides=2),
            # Lambda(self._print),
            self.get_generator_block(im_chan, kernel_size=4, strides=2, final_layer=True),
            # Lambda(self._print)
        ])

    @staticmethod
    def get_generator_block(output_channels, kernel_size=3, strides=2, final_layer=False):
        """
        Function for returning a block of the generator's neural network

        Parameters:
            output_channels: the dimension of the output channels
            kernel_size: the size of the kernel
            strides: the strides
            final_layer: whether the block is the last layer

        Returns:
            a generator neural network layer
        """
        if not final_layer:
            return tf.keras.Sequential(layers=[
                Conv2DTranspose(output_channels, kernel_size, strides=strides, padding="valid"),
                BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True),
                Activation("relu")
            ])

        else:
            return tf.keras.Sequential(layers=[
                Conv2DTranspose(output_channels, kernel_size, strides=strides, padding="valid"),
                Activation("tanh")
            ])

    @staticmethod
    def get_noise(n_samples, z_dim):
        """
        Get the noise

        Parameters:
            n_samples: The number of samples
            z_dim: the dimension of noise

        Returns:
            a noise tensor
        """
        return np.random.randn(n_samples, 1, 1, z_dim)

    @staticmethod
    def _print(x):
        """
        Print tensor

        Parameters:
            x: tensor

        Returns:
            tensor with printing operation
        """
        tf.print(tf.shape(x))

        return x
