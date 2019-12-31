"""Model constructors for use in notebooks."""
import numpy as np
import tensorflow as tf

from math import ceil


def dense(inputs, weights, bias, activation=tf.identity, *activation_args):
    """Fully-connected layer."""
    x = tf.matmul(inputs, weights) + bias
    return activation(x, *activation_args)


def conv2d(inputs, filters, strides=1, padding="SAME",
           activation=tf.nn.relu, *activation_args):
    """Convolutional layer with 2D filter and ReLU activation."""
    x = tf.nn.conv2d(inputs, filters, strides, padding)
    return activation(x, *activation_args)


def max_pool2d(inputs, ksize, strides=2, padding="SAME"):
    """Max pooling layer."""
    x = tf.nn.max_pool2d(inputs, ksize, strides, padding)
    return x


def init_weights(shape, initializer=tf.initializers.glorot_uniform()):
    """Initialize weights for tensorflow layer."""
    weights = tf.Variable(
        initializer(shape),
        trainable=True,
        dtype=tf.float32
    )

    return weights


class SiameseNetwork(object):
    """Siamese CNN for facial recognition."""

    def __init__(self,
                 input_dim=(60, 60),
                 embed_dim=1024,
                 optimizer=tf.optimizers.Adam,
                 checkpoint_dir=None,
                 **optimizer_kwargs):
        """Set hyperparameters, initialize weights."""
        self.input_dim = input_dim  # the input will be resized to this
        self.optimizer = optimizer(**optimizer_kwargs)

        np.random.seed(41)
        self.initialize_weights(embed_dim)

        if checkpoint_dir is not None:
            assert checkpoint_dir[-1] == "/"
            self.checkpoint_dir = checkpoint_dir
            weights = {"w" + str(i): w for (i, w) in enumerate(self.weights)}
            self.checkpoint = tf.train.Checkpoint(
                optimizer=self.optimizer,
                **weights
            )

    def initialize_weights(self, embed_dim):
        """Initialize weights."""
        x = self.input_dim[0]
        y = self.input_dim[1]

        wshapes = [
            [5, 5, 1, 128],
            [3, 3, 128, 256],
            [256 * ceil(x / 4) * ceil(y / 4), embed_dim],
            [embed_dim, 1]
        ]

        bshapes = [
            [1, embed_dim]
        ]

        rw = range(len(wshapes))
        weights = [init_weights(wshapes[i]) for i in rw]
        self.weights = weights

        rb = range(len(bshapes))
        bias_initializer = tf.initializers.zeros()
        biases = [init_weights(bshapes[i], bias_initializer) for i in rb]

        self.biases = biases

    def embed(self, x):
        """Tensorflow model function."""
        x = tf.cast(x, dtype=tf.float32)
        x = tf.image.resize(x, self.input_dim)

        c1 = conv2d(x, self.weights[0])
        p1 = max_pool2d(c1, 2)

        c2 = conv2d(p1, self.weights[1])
        p2 = max_pool2d(c2, 2)

        flatten = tf.reshape(p2, shape=(tf.shape(p2)[0], -1))
        embed = dense(flatten, self.weights[2], self.biases[0], tf.nn.tanh)
        return embed

    def similarity_score(self, x1, x2, embed=True, logits=True):
        """Compute similarity score between -1 and 1."""
        embed1 = self.embed(x1) if embed else x1
        embed2 = self.embed(x2) if embed else x2

        dist = tf.abs(embed2 - embed1)

        score = dense(dist, self.weights[3], tf.constant([0.]))

        if logits:
            return score
        else:
            return tf.nn.sigmoid(score)

    def loss(self, labels, preds):
        """Cross entropy loss."""
        return tf.losses.binary_crossentropy(labels, preds, True)

    def train_step(self, x1, x2, labels):
        """One train step on a batch of inputs and target outputs."""
        with tf.GradientTape() as tape:
            preds = self.similarity_score(x1, x2)
            current_loss = self.loss(labels, preds)

        grads = tape.gradient(current_loss, self.weights + self.biases)
        self.optimizer.apply_gradients(zip(grads, self.weights + self.biases))
        loss = tf.reduce_mean(current_loss)

        return grads, loss

    def save(self, name=""):
        """Save checkpoint."""
        path = self.checkpoint_dir + name
        self.checkpoint.save(path)
        print("Checkpoint saved to", path)

    def load(self, name):
        """Load checkpoint."""
        path = self.checkpoint_dir + name
        self.checkpoint.restore(path)
        print("Checkpoint loaded from", path)
