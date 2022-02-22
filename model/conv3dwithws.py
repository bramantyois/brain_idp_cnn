import tensorflow as tf
from tensorflow.keras.layers import Conv3d


class Conv3dWithWeightStandardization(Conv3d):
    def standardize_weight(self, weights, eps=1e-6):
        mean = tf.math.reduce_mean(weights, axis=(0, 1, 2), keepdims=True)
        var = tf.math.reduce_variance(weights, axis=(0, 1, 2), keepdims=True)
        fan_in = np.prod(weight.shape[:-1])
        gain = self.add_weight(
            name="gain",
            shape=(weight.shape[-1],),
            initializer="ones",
            trainable=True,
            dtype=self.dtype,
        )
        scale = (
            tf.math.rsqrt(
                tf.math.maximum(var * fan_in, tf.convert_to_tensor(eps, dtype=self.dtype))
            )
            * gain
        )
        return weight * scale - (mean * scale)