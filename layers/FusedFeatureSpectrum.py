# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class FusedFeatureSpectrum(layers.Layer):
    def __init__(self, **kwargs):
        super(FusedFeatureSpectrum, self).__init__(**kwargs)

    def call(self, attention_feature_spectrum, conv_spectrum):
        new_afs_shape = list(tf.shape(attention_feature_spectrum).numpy()) + [1]
        new_afs = tf.reshape(attention_feature_spectrum, shape=new_afs_shape)

        return tf.add(conv_spectrum, tf.multiply(conv_spectrum, new_afs))
