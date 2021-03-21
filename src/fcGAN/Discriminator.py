import tensorflow as tf
from tensorflow.keras.layers import Dense, LeakyReLU, Activation


class Discriminator:

    def __init__(self, hidden_dim=128):
        """
        Initialize a discriminator

        Parameters:
            hidden_dim: the dimension of hidden layers
        """
        self.disc = tf.keras.Sequential(layers=[
            self.get_discriminator_block(hidden_dim * 4),
            self.get_discriminator_block(hidden_dim * 2),
            self.get_discriminator_block(hidden_dim),
            Dense(1),
            Activation("sigmoid")
        ])

    @staticmethod
    def get_discriminator_block(output_dim):
        """
        Discriminator Block
        Function for returning a neural network of the discriminator given input and output dimensions.

        Parameters:
            output_dim: the dimension of the output vector, a scalar

        Returns:
            a discriminator neural network layer, with a linear transformation
              followed by an nn.LeakyReLU activation with negative slope of 0.2
        """
        return tf.keras.Sequential(layers=[
            Dense(output_dim),
            LeakyReLU(alpha=0.2)
        ])