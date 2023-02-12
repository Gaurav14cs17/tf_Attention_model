import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv1D


class ChannelAttention2D(tf.keras.layers.Layer):
    def __init__(self, nf, r=4, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.conv1 = Conv2D(filters=nf / r, kernel_size=1, use_bias=True)
        self.conv2 = Conv2D(filters=nf, kernel_size=1, use_bias=True)

    @tf.function
    def call(self, x):
        y = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        y = self.conv1(y)
        y = tf.nn.relu(y)
        y = self.conv2(y)
        y = tf.nn.sigmoid(y)
        y = tf.multiply(x, y)
        return y

    def get_config(self):
        config = super().get_config()
        config.update({"Att_filters": self.nf})
        return config


class EfficientChannelAttention2D(tf.keras.layers.Layer):
    def __init__(self, nf, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.conv1 = Conv1D(filters=1, kernel_size=3, activation=None, padding="same", use_bias=False)

    @tf.function
    def call(self, x):
        pool = tf.reduce_mean(x, [1, 2])
        pool = tf.expand_dims(pool, -1)
        att = self.conv1(pool)  # set k=3 for every channel size between 8 and 64
        att = tf.transpose(att, perm=[0, 2, 1])
        att = tf.expand_dims(att, 1)
        att = tf.sigmoid(att)
        y = tf.multiply(x, att)
        return y

    def get_config(self):
        config = super().get_config()
        config.update({"Att_filters": self.nf})
        return config


class PixelAttention2D(tf.keras.layers.Layer):
    def __init__(self, nf, **kwargs):
        super().__init__(**kwargs)
        self.nf = nf
        self.conv1 = Conv2D(filters=nf, kernel_size=1)

    @tf.function
    def call(self, x):
        y = self.conv1(x)
        self.sig = tf.keras.activations.sigmoid(y)
        out = tf.math.multiply(x, y)
        out = self.conv1(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"Att_filters": self.nf})
        return config



if __name__ == '__main__':
    inpt_1 = tf.random.normal((1 , 255 , 255 , 3))
    cnn_layer = Conv2D(32,3,activation='relu', padding='same')(inpt_1)

    # Using the .shape[-1] to simplify network modifications. Can directly input number of channels as well
    Pixel_attention_cnn = PixelAttention2D(cnn_layer.shape[-1])(cnn_layer)
    Channel_attention_cnn = ChannelAttention2D(cnn_layer.shape[-1])(cnn_layer)
    EfficientChannelAttention_cnn = EfficientChannelAttention2D(cnn_layer.shape[-1])(cnn_layer)
    print(Pixel_attention_cnn.shape , Channel_attention_cnn.shape , EfficientChannelAttention_cnn.shape)