import tensorflow as tf
from tensorflow.keras.layers import Conv3D


class Conv3DWithWeightStandardization(Conv3D):
    def standardize_weight(self, kernel, std_tol=1e-6):
        mean = tf.reduce_mean(kernel, axis=(0,1,2,3), keepdims=True)
        std = tf.math.reduce_std(kernel, axis=(0,1,2,3), keepdims=True)
        std = tf.math.maximum(std, std_tol)

        return (kernel-mean) / std

    def call(self, inputs):
        self.kernel.assign(self.standardize_weight(self.kernel))
        return super().call(inputs)
