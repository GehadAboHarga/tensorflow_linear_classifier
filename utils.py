
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def get_weights(n_features, n_labels):
    """
    Return TensorFlow weights
    :param n_features: Number of features
    :param n_labels: Number of labels
    :return: TensorFlow weights
    """
    weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
    #  Return weights
    return weights


def get_biases(n_labels):
    """
    Return TensorFlow bias
    :param n_labels: Number of labels
    :return: TensorFlow bias
    """
    bias = tf.Variable(tf.zeros(n_labels))
    #  Return biases
    return bias


def linear(input, w, b):
    """
    Return linear function in TensorFlow
    :param input: TensorFlow input
    :param w: TensorFlow weights
    :param b: TensorFlow biases
    :return: TensorFlow linear function
    """
    linearf = tf.add(tf.matmul(input,w),b )
    #  Linear Function (xW + b)
    return linearf
