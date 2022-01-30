# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class FusedFeatureSpectrum(layers.Layer):
    def __init__(self, **kwargs):
        super(FusedFeatureSpectrum, self).__init__(**kwargs)

    def call(self, attention_feature_spectrum, conv_spectrum):
        # new_afs = tf.expand_dims(attention_feature_spectrum, axis=-1)

        return tf.add(conv_spectrum, attention_feature_spectrum) # tf.multiply(conv_spectrum, new_afs))
