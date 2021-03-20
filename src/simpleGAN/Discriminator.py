import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU


class Discriminator:

    def __init__(self, im_dim=784, hidden_dim=128):
        """
        Initialize a discriminator

        Parameters:
            im_dim: the dimension of input image
            hidden_dim: the dimension of hidden layers
        """
        self.disc = tf.keras.Sequential(layers=[
            self.get_discriminator_block(im_dim, hidden_dim * 4),
            self.get_discriminator_block(hidden_dim * 4, hidden_dim * 2),
            self.get_discriminator_block(hidden_dim * 2, hidden_dim),
            Dense(hidden_dim, 1),
        ])

    @staticmethod
    def get_discriminator_block(input_dim, output_dim):
        """
        Discriminator Block
        Function for returning a neural network of the discriminator given input and output dimensions.

        Parameters:
            input_dim: the dimension of the input vector, a scalar
            output_dim: the dimension of the output vector, a scalar

        Returns:
            a discriminator neural network layer, with a linear transformation
              followed by an nn.LeakyReLU activation with negative slope of 0.2
        """
        return tf.keras.Sequential(layers=[
            Dense(input_dim, output_dim),
            LeakyReLU(alpha=0.2)
        ])