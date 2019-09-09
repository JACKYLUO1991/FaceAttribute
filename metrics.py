from keras import backend as K
import tensorflow as tf

# 117/101/71 UTKFace dataset/other datasets/megaage_asian dataset
k = 71


def mae(y_true, y_pred, k=k):
    true_age = K.sum(y_true * K.arange(0, k, dtype="float32"), axis=-1)
    pred_age = K.sum(y_pred * K.arange(0, k, dtype="float32"), axis=-1)
    mae = K.mean(K.abs(true_age - pred_age))
    return mae


# def focal_loss_multi(gamma=2.):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         return -K.sum(K.pow(1. - pt_1, gamma) * K.log(pt_1))
#     return focal_loss_fixed


def focal_loss(gamma=2, alpha=0.4):
    """Focal loss"""
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed
