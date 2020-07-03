import tensorflow as tf


def weighted_dice_coeff(y_true, y_pred, smooth=1, weights=None):
    intersection = y_true * y_pred

    score = (tf.reduce_sum(2. * intersection * weights) + smooth) / \
        (tf.reduce_sum((y_true + y_pred) * weights) + smooth)
    return tf.reduce_mean(score)


def dice_coeff(y_true, y_pred, smooth=1):
    intersection = y_true * y_pred

    score = (tf.reduce_sum(2. * intersection) + smooth) / \
        (tf.reduce_sum((y_true + y_pred)) + smooth)
    return tf.reduce_mean(score)


def focal_loss(y_true, y_pred, gamma=4, beta=0, weights=[1, 1, 1, 1]):
    pt = tf.math.sigmoid(gamma * ((y_true - 0.5)*2)
                         * ((y_pred - 0.5)*2) + beta)
    loss = -  tf.math.log(pt) / gamma
    return tf.reduce_mean(loss * weights)


def superloss(y_true, y_pred, gamma=1):
    weights = tf.reduce_sum(y_true, axis=(0, 1, 2)) + 1e-6
    weights = weights / (tf.reduce_sum(weights) + 1e-6)
    return tf.pow(focal_loss(y_true, y_pred, 4, 0, weights=weights), gamma) + tf.pow(- tf.math.log(weighted_dice_coeff(y_true, y_pred, weights=weights)), gamma)
