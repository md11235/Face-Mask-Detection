# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ChannelWiseDotProduct(layers.Layer):
    def __init__(self, **kwargs):
        super(ChannelWiseDotProduct, self).__init__(**kwargs)

    def call(self, attention_spectrum, conv_spectrum):
        tensor_a = attention_spectrum
        tensor_b = conv_spectrum

        # return tf.reduce_sum(layers.multiply([tensor_a, tensor_b]), -1)
        return layers.multiply([tensor_a, tensor_b])
