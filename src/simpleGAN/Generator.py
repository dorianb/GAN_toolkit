import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Activation


class Generator:

    def _print(self, x):
        tf.print(x)
        return x

    def __init__(self, im_dim=784, hidden_dim=128):
        """
        Initialize a generator

        Parameters:
             im_dim: the image dimension
             hidden_dim: the hidden layers dimension
        """
        self.gen = tf.keras.Sequential(layers=[
            self.get_generator_block(hidden_dim),
            self.get_generator_block(hidden_dim * 2),
            self.get_generator_block(hidden_dim * 4),
            self.get_generator_block(hidden_dim * 8),
            Dense(im_dim),
            Activation("sigmoid")
        ])

    @staticmethod
    def get_generator_block(output_dim):
        """
        Function for returning a block of the generator's neural network
        given input and output dimensions.

        Parameters:
            output_dim: the dimension of the output vector, a scalar

        Returns:
            a generator neural network layer, with a linear transformation
              followed by a batch normalization and then a relu activation
        """
        return tf.keras.Sequential(layers=[
            Dense(output_dim),
            BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True),
            ReLU()
        ])
