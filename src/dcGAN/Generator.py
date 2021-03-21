import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, BatchNormalization, Activation


class Generator:

    def __init__(self, im_chan=1, hidden_dim=64):
        """
        Initialize a generator

        Parameters:
             im_chan: the image channel dimension
             hidden_dim: the hidden layers dimension
        """
        self.gen = tf.keras.Sequential(layers=[
            self.get_generator_block(hidden_dim * 4, kernel_size=3, strides=2),
            self.get_generator_block(hidden_dim * 2, kernel_size=4, strides=1),
            self.get_generator_block(hidden_dim, kernel_size=3, strides=2),
            self.get_generator_block(im_chan, kernel_size=4, strides=2, final_layer=True),
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
                Conv2DTranspose(output_channels, kernel_size, strides=strides),
                BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True),
                Activation("relu")
            ])

        else:
            return tf.keras.Sequential(layers=[
                Conv2DTranspose(output_channels, kernel_size, strides=strides),
                Activation("tanh")
            ])
