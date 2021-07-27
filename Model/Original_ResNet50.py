from keras import Model
from keras.layers import Add, Conv2D, MaxPooling2D, GlobalAveragePooling2D, ReLU, BatchNormalization, Input, \
    Dense
from tensorflow import Tensor
from keras.initializers import he_normal
from Attention.AttentionModule import SENet, CBAM


def stage_0(inputs: Tensor):
    x = Conv2D(
        kernel_size=7,
        padding='same',
        strides=2,
        filters=64,
        kernel_initializer=he_normal()
    )(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(
        pool_size=3,
        padding='same',
        strides=2
    )(x)
    return x


def Stage_1_1st_ResBlock(inputs: Tensor, cbam: bool = False, reduction: int = 16, senet: bool = False):
    add_1 = Conv2D(
        filters=64,
        kernel_size=1,
        padding='same',
        kernel_initializer=he_normal()
    )(inputs)
    add_2 = Conv2D(
        filters=64 * 4,
        kernel_size=1,
        padding='same',
        kernel_initializer=he_normal()
    )(inputs)

    add_1 = BatchNormalization()(add_1)
    add_1 = ReLU()(add_1)

    add_1 = Conv2D(
        kernel_size=3,
        filters=64,
        padding='same',
        kernel_initializer=he_normal()
    )(add_1)
    add_1 = BatchNormalization()(add_1)
    add_1 = ReLU()(add_1)

    add_1 = Conv2D(
        kernel_size=1,
        filters=256,
        padding='same',
        kernel_initializer=he_normal()
    )(add_1)
    add_1 = BatchNormalization()(add_1)

    if cbam:
        add_1 = CBAM(add_1, reduction=reduction)
        senet = False
    if senet:
        add_1 = SENet(add_1, reduction=reduction)

    add = Add()([add_1, add_2])

    out = ReLU()(add)
    return out


def ResBlock(inputs: Tensor, filters: int, initial_block: bool = False, cbam: bool = False, reduction: int = 16,
             senet: bool = False):
    if initial_block:
        add_1 = Conv2D(
            kernel_size=1,
            padding='same',
            filters=filters,
            strides=2,
            kernel_initializer=he_normal()
        )(inputs)
        add_2 = Conv2D(
            kernel_size=1,
            padding='same',
            filters=filters * 4,
            strides=2,
            kernel_initializer=he_normal()
        )(inputs)
    else:
        add_1 = Conv2D(
            kernel_size=1,
            padding='same',
            filters=filters,
            strides=1,
            kernel_initializer=he_normal()
        )(inputs)
        add_2 = inputs

    add_1 = BatchNormalization()(add_1)
    add_1 = ReLU()(add_1)

    add_1 = Conv2D(
        kernel_size=3,
        filters=filters,
        padding='same',
        kernel_initializer=he_normal()
    )(add_1)
    add_1 = BatchNormalization()(add_1)
    add_1 = ReLU()(add_1)

    add_1 = Conv2D(
        kernel_size=1,
        filters=filters * 4,
        padding='same',
        kernel_initializer=he_normal()
    )(add_1)

    add_1 = BatchNormalization()(add_1)
    if cbam:
        add_1 = CBAM(add_1, reduction=reduction)
        senet = False
    if senet:
        add_1 = SENet(add_1, reduction=reduction)

    add = Add()([add_1, add_2])

    out = ReLU()(add)
    return out


def mkResNet50(inputs_shape=None, classes_num: int = 6, senet: bool = False, CBAM: bool = False, reduction: int = 16):
    if inputs_shape is None:
        inputs_shape = (224, 224, 3)
    input_ = Input(inputs_shape)
    s0 = stage_0(input_)
    s1 = Stage_1_1st_ResBlock(s0, senet=senet, cbam=CBAM, reduction=reduction)
    for i in range(1, 3):
        s1 = ResBlock(s1, 64, False, senet=senet, cbam=CBAM, reduction=reduction)
    s2 = ResBlock(s1, 128, True, senet=senet, cbam=CBAM, reduction=reduction)
    for i in range(1, 4):
        s2 = ResBlock(s2, 128, False, senet=senet, cbam=CBAM, reduction=reduction)
    s3 = ResBlock(s2, 256, True, senet=senet, cbam=CBAM, reduction=reduction)
    for i in range(1, 6):
        s3 = ResBlock(s3, 256, False, senet=senet, cbam=CBAM, reduction=reduction)
    s4 = ResBlock(s3, 512, True, senet=senet, cbam=CBAM, reduction=reduction)
    for i in range(1, 3):
        s4 = ResBlock(s4, 512, False, senet=senet, cbam=CBAM, reduction=reduction)
    gap = GlobalAveragePooling2D(name='gap')(s4)
    output = Dense(activation='softmax', name='softmax', units=classes_num)(gap)
    model = Model(inputs=input_, outputs=output)
    return model


if __name__ == '__main__':
    import sys

    model = mkResNet50(classes_num=6, senet=True)
    model.summary()
    sys.exit()
