import math
from functools import partial
from typing import Union
import tensorflow as tf
from tensorflow.keras import layers, Model


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def correct_pad(input_size: Union[int, tuple], kernel_size: int):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    Arguments:
      input_size: Input tensor size.
      kernel_size: An integer or tuple/list of 2 integers.

    Returns:
      A tuple.
    """
    if isinstance(input_size, int):
        input_size = (input_size, input_size)
    kernel_size = (kernel_size, kernel_size)

    adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
    correct = (kernel_size[0] // 2, kernel_size[1] // 2)
    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


class HardSigmoid(layers.Layer):
    def __init__(self, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        self.relu6 = layers.ReLU(max_value=6.0)

    def call(self, inputs, *args, **kwargs):
        x = self.relu6(inputs + 3) * (1.0 / 6)
        return x


class HardSwish(layers.Layer):
    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)
        self.hard_sigmoid = HardSigmoid()

    def call(self, inputs, *args, **kwargs):
        x = self.hard_sigmoid(inputs) * inputs
        return x


def RMA_attention(inputs, b=1, gamma=2):
    """
        平行注意力机制
    Args:
        inputs:
        b:
        gamma:

    Returns:

    """
    input_channel = inputs.shape[-1]
    kernel_size = int(abs((math.log(input_channel, 2) + b) / gamma))
    kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

    x_max_c = layers.GlobalMaxPool2D()(inputs)  # x: c
    x_max_c = tf.expand_dims(x_max_c, 2)  # x: c, 1

    x_avg_c = layers.GlobalAvgPool2D()(inputs)  # x: c
    x_avg_c = tf.expand_dims(x_avg_c, 2)  # x: c, 1

    x_con_c = layers.Concatenate()([x_max_c, x_avg_c])  # x: c, 2
    # print(x_con.shape)
    x_con_c = layers.Conv1D(1,
                            kernel_size=kernel_size,
                            padding="same",
                            use_bias=False)(x_con_c)  # x: c, 1
    x_con_c = layers.Activation('sigmoid')(x_con_c)
    x_chan = layers.Reshape((1, 1, -1))(x_con_c)  # x: c,1 -> 1,1,c

    x_max_s = tf.reduce_max(inputs, 3, keepdims=True)
    x_avg_s = tf.reduce_mean(inputs, 3, keepdims=True)
    x_con_s = layers.Concatenate()([x_max_s, x_avg_s])
    x_con_s = layers.Conv2D(1, kernel_size=7, padding='same')(x_con_s)
    x_spti = layers.Activation("sigmoid")(x_con_s)

    x_mult = layers.Multiply()([inputs, x_chan, x_spti])
    outputs = layers.Add()([inputs, x_mult])
    return outputs


def inverted_residual_block(
        x,
        input_channel: int,
        kernel_size: int,
        expand_channel: int,
        out_channel: int,
        use_attention: bool,
        activation: str,
        stride: int,
        block_id: int,
        alpha: float = 1.0
):
    bn = partial(layers.BatchNormalization, epsilon=1e-5, momentum=0.99)

    input_channel = _make_divisible(input_channel * alpha)
    expand_channel = _make_divisible(expand_channel * alpha)
    out_channel = _make_divisible(out_channel * alpha)

    activation = layers.ReLU if activation == "RE" else HardSwish

    shortcut = x
    prefix = "expanded_conv"
    if block_id:
        # expand channel
        prefix = "expanded_conv_{}".format(block_id)
        x = layers.Conv2D(
            filters=expand_channel,
            kernel_size=1,
            padding="same",
            use_bias=False,
            name=prefix + "expand"
        )(x)
        x = bn(name=prefix + "expand/BatchNorm")(x)
        x = activation(name=prefix + 'expand/' + activation.__name__)(x)

    if stride == 2:
        input_size = (x.shape[1], x.shape[2])
        x = layers.ZeroPadding2D(
            padding=correct_pad(input_size, kernel_size),
            name=prefix + 'depthwise/pad'
        )(x)

    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=stride,
        padding='same' if stride == 1 else 'valid',
        use_bias=False,
        name=prefix + 'depthwise'
    )(x)
    x = bn(name=prefix + 'depthwise/BatchNorm')(x)
    x = activation(name=prefix + 'depthwise/' + activation.__name__)(x)

    if use_attention:
        x = RMA_attention(x)

    x = layers.Conv2D(
        filters=out_channel,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name=prefix + "project"
    )(x)
    x = bn(name=prefix + 'project/BatchNorm')(x)

    if stride == 1 and input_channel == out_channel:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])

    return x


def RMA_mobile(input_shape=(224, 224, 3),
               num_classes=1000,
               alpha=1.0,
               include_top=True):
    bn = partial(layers.BatchNormalization, epsilon=0.001, momentum=0.99)
    img_input = layers.Input(shape=input_shape)

    # out -> (112, 112, filters)
    x = layers.Conv2D(filters=16,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='same',
                      use_bias=False,
                      name="Conv")(img_input)
    x = bn(name="Conv/BatchNorm")(x)
    x = HardSwish(name="Conv/HardSwish")(x)

    inverted_cnf = partial(inverted_residual_block, alpha=alpha)
    # input, input_c, k_size, expand_c, use_se, activation, stride, block_id
    x = inverted_cnf(x, 16, 3, 16, 16, True, "RE", 2, 0)
    x = inverted_cnf(x, 16, 3, 72, 24, False, "RE", 2, 1)
    x = inverted_cnf(x, 24, 3, 88, 24, False, "RE", 1, 2)
    x = inverted_cnf(x, 24, 5, 96, 40, True, "HS", 2, 3)
    x = inverted_cnf(x, 40, 5, 240, 40, True, "HS", 1, 4)
    x = inverted_cnf(x, 40, 5, 240, 40, True, "HS", 1, 5)
    x = inverted_cnf(x, 40, 5, 120, 48, True, "HS", 1, 6)
    x = inverted_cnf(x, 48, 5, 144, 48, True, "HS", 1, 7)
    x = inverted_cnf(x, 48, 5, 288, 96, True, "HS", 2, 8)
    x = inverted_cnf(x, 96, 5, 576, 96, True, "HS", 1, 9)
    x = inverted_cnf(x, 96, 5, 576, 96, True, "HS", 1, 10)

    last_c = _make_divisible(96 * 6 * alpha)
    last_point_c = _make_divisible(1024 * alpha)

    x = layers.Conv2D(filters=last_c,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      name="Conv_1")(x)
    x = bn(name="Conv_1/BatchNorm")(x)
    x = HardSwish(name="Conv_1/HardSwish")(x)

    if include_top is True:
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape((1, 1, last_c))(x)

        # fc1
        x = layers.Conv2D(filters=last_point_c,
                          kernel_size=1,
                          padding='same',
                          name="Conv_2")(x)
        x = HardSwish(name="Conv_2/HardSwish")(x)

        # fc2
        x = layers.Conv2D(filters=num_classes,
                          kernel_size=1,
                          padding='same',
                          name='Logits/Conv2d_1c_1x1')(x)
        x = layers.Flatten()(x)
        x = layers.Softmax(name="Predictions")(x)

    model = Model(img_input, x, name="RMA_mobile")
    return model
