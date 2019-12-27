"""Model constructors for use in notebooks."""
import tensorflow as tf


def dense(inputs, weights, alpha=0.2):
    """Fully-connected layer."""
    x = tf.matmul(inputs, weights)
    return tf.nn.leaky_relu(x, alpha=alpha)


def conv2d(inputs, filters, strides=1, padding="SAME", alpha=0.2):
    """Convolutional layer with 2D filter and ReLU activation."""
    x = tf.nn.conv2d(inputs, filters, strides, padding)
    return tf.nn.leaky_relu(x, alpha=alpha)


def max_pool2d(inputs, ksize, strides=2, padding="SAME"):
    """Max pooling layer."""
    x = tf.nn.max_pool2d(inputs, ksize, strides, padding)
    return x


def init_weights(shape, name,
                 initializer=tf.initializers.glorot_uniform()):
    """Initialize weights for tensorflow layer."""
    weights = tf.Variable(
        initializer(shape),
        name=name,
        trainable=True,
        dtype=tf.float32
    )

    return weights


class SiameseNetwork(object):
    """Siamese CNN for facial recognition."""

    def __init__(self,
                 input_dim=(243, 320),
                 embed_dim=64,
                 optimizer=tf.optimizers.Adam,
                 learning_rate=0.001,
                 checkpoint_dir=None):
        """Set hyperparameters, initialize weights."""
        self.optimizer = optimizer(learning_rate)

        self.initialize_weights(input_dim, embed_dim)

        if checkpoint_dir is not None:
            assert checkpoint_dir[-1] == "/"
            self.checkpoint_dir = checkpoint_dir
            weights = {"w" + str(i): w for (i, w) in enumerate(self.weights)}
            self.checkpoint = tf.train.Checkpoint(
                optimizer=self.optimizer,
                **weights
            )

    def initialize_weights(self, input_dim, embed_dim):
        """Initialize weights."""
        shapes = [
            [5, 5, 1, 32],
            [5, 5, 32, 32],
            [5, 5, 32, 64],
            [5, 5, 64, 64],
            [64 * input_dim[0] * input_dim[1] / 16, embed_dim]
        ]

        r = range(len(shapes))
        weights = [init_weights(shapes[i], "weight{}".format(i)) for i in r]
        self.weights = weights

    def embed(self, x):
        """Tensorflow model function."""
        x = tf.cast(x, dtype=tf.float32)
        c1_1 = conv2d(x, self.weights[0])
        c1_2 = conv2d(c1_1, self.weights[1])
        p1 = max_pool2d(c1_2, 2)

        c2_1 = conv2d(p1, self.weights[2])
        c2_2 = conv2d(c2_1, self.weights[3])
        p2 = max_pool2d(c2_2, 2)

        flatten = tf.reshape(p2, shape=(tf.shape(p2)[0], -1))
        embed = dense(flatten, self.weights[4])
        return embed

    def similarity_score(self, x1, x2):
        """Compute similarity score between -1 and 1."""
        embed1 = self.embed(x1)
        embed2 = self.embed(x2)

        dot_product = tf.matmul(embed1, tf.transpose(embed2))
        norms = tf.norm(embed1) * tf.norm(embed2)
        cosine_similarity = dot_product / norms
        return cosine_similarity

    def loss(self, labels, preds):
        """Cross entropy loss."""
        return tf.losses.categorical_crossentropy(labels, preds)

    def train_step(self, x1, x2, labels):
        """One train step on a batch of inputs and target outputs."""
        with tf.GradientTape() as tape:
            preds = self.similarity_score(x1, x2)
            current_loss = self.loss(labels, preds)

        grads = tape.gradient(current_loss, self.weights)
        self.optimizer.apply_gradients(zip(grads, self.weights))

        loss = tf.reduce_mean(current_loss).numpy()

        return loss

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
