import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Activation


class Generator:

    def __init__(self, z_dim=10, im_dim=784, hidden_dim=128):
        """
        Initialize a generator

        Parameters:
             z_dim: the noise vector dimension
             im_dim: the image dimension
             hidden_dim: the hidden layers dimension
        """
        self.gen = tf.keras.Sequential(
            self.get_generator_block(z_dim, hidden_dim),
            self.get_generator_block(hidden_dim, hidden_dim * 2),
            self.get_generator_block(hidden_dim * 2, hidden_dim * 4),
            self.get_generator_block(hidden_dim * 4, hidden_dim * 8),
            Dense(hidden_dim * 8, im_dim),
            Activation("sigmoid")
        )

    @staticmethod
    def get_generator_block(input_dim, output_dim):
        """
        Function for returning a block of the generator's neural network
        given input and output dimensions.

        Parameters:
            input_dim: the dimension of the input vector, a scalar
            output_dim: the dimension of the output vector, a scalar

        Returns:
            a generator neural network layer, with a linear transformation
              followed by a batch normalization and then a relu activation
        """
        return tf.keras.Sequential(layers=[
            Dense(input_dim, output_dim),
            BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True),
            ReLU()
        ])
