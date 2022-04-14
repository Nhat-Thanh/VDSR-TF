from tensorflow.keras.layers import Conv2D, Input, Activation, Add
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.models import Model
import tensorflow as tf


def VDSR_model():
    X_in = Input(shape=(None, None, 3))
    X = X_in

    for _ in range(0, 20):
        X = Conv2D(64, 3, padding='same', kernel_initializer=HeNormal())(X)
        X = Activation('relu')(X)
    X = Conv2D(3, 3, padding='same', kernel_initializer=HeNormal())(X)
    X = Add()([X, X_in])

    X_out = tf.clip_by_value(X, 0.0, 1.0)

    return Model(X_in, X_out)
