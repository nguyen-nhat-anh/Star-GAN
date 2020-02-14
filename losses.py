import tensorflow as tf


def adversarial_loss(src_fake, src_real=None):
    """Wasserstein GAN loss"""
    loss_fake = -tf.reduce_mean(src_fake)
    if src_real is not None:
        loss_real = tf.reduce_mean(src_real)
        return loss_real + loss_fake
    return loss_fake


def classification_loss(target, logits):
    return tf.keras.losses.BinaryCrossentropy(from_logits=True)(target, logits)


def reconstruction_loss(img_real, img_rec):
    """Cycle consistency loss"""
    return tf.reduce_mean(tf.abs(img_real - img_rec))