import sys
from keras.utils import plot_model
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, ReLU, Concatenate, Add, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow import Tensor

from Model.GroupConv2D import GroupConv2D


def stage_0(input: Tensor):
    conv00 = Conv2D(
        filters=64,
        kernel_size=7,
        padding='same',
        strides=2
    )(input)
    BN01 = BatchNormalization()(conv00)
    RL00 = ReLU()(BN01)
    # MP00 = MaxPooling2D(3, 2, padding='same')(RL00)
    return RL00


def stage_1_ResBlocK(input: Tensor, kernels=None, groups=None, filters=None, initial_block: bool = False
                     ):
    if kernels is None:
        kernels = [9, 7, 5, 3]
    if filters is None:
        filters = 16
    if groups is None:
        groups = [16, 8, 4, 1]

    if initial_block:
        # Move the DownSample to the first block of each stage
        input = MaxPooling2D(
            pool_size=3,
            padding='same',
            strides=2
        )(input)

    #
    conv_10 = Conv2D(
        filters=64,
        padding='same',
        kernel_size=1
    )(input)
    BN_0 = BatchNormalization()(conv_10)
    relu_0 = ReLU()(BN_0)

    # PyConv

    GC_0 = GroupConv2D(out_filters=filters,
                       kernel_size=kernels[0],
                       groups=groups[0],
                       padding='same'
    )(relu_0)

    GC_1 = GroupConv2D(out_filters=filters,
                       kernel_size=kernels[1],
                       groups=groups[1],
                       padding='same'
    )(relu_0)

    GC_2 = GroupConv2D(out_filters=filters,
                       kernel_size=kernels[2],
                       groups=groups[2],
                       padding='same'
    )(relu_0)

    GC_3 = GroupConv2D(out_filters=filters,
                       kernel_size=kernels[3],
                       groups=groups[3],
                       padding='same'
    )(relu_0)

    GC_PyConv = Concatenate(axis=-1)([GC_0, GC_1, GC_2, GC_3])
    BN_1 = BatchNormalization()(GC_PyConv)
    relu_1 = ReLU()(BN_1)
    conv_1 = Conv2D(
        filters=256,
        padding='same',
        kernel_size=1
    )(relu_1)

    if initial_block:
        x = Conv2D(
            filters=256,
            kernel_size=1,
            padding='same'
        )(input)
        x = BatchNormalization()(x)
    else:
        x = input

    out = Add()([conv_1, x])
    return out


def stage_2_ResBlocK(input: Tensor, kernels=None, groups=None, filters=None, initial_block: bool = False
                     ):
    if kernels is None:
        kernels = [7, 5, 3]
    if filters is None:
        filters = [64, 32, 32]
    if groups is None:
        groups = [8, 4, 1]

    if initial_block:
        # Move the DownSample to the first block of each stage
        input = MaxPooling2D(
            pool_size=3,
            padding='same',
            strides=2
        )(input)

    #
    conv_10 = Conv2D(
        filters=128,
        padding='same',
        kernel_size=1
    )(input)
    BN_0 = BatchNormalization()(conv_10)
    relu_0 = ReLU()(BN_0)

    # PyConv

    GC_0 = GroupConv2D(out_filters=filters[0],
                       kernel_size=kernels[0],
                       groups=groups[0],
                       padding='same'
    )(relu_0)

    GC_1 = GroupConv2D(out_filters=filters[1],
                       kernel_size=kernels[1],
                       groups=groups[1],
                       padding='same'
    )(relu_0)

    GC_2 = GroupConv2D(out_filters=filters[2],
                       kernel_size=kernels[2],
                       groups=groups[2],
                       padding='same'
    )(relu_0)

    GC_PyConv = Concatenate(axis=-1)([GC_0, GC_1, GC_2])
    BN_1 = BatchNormalization()(GC_PyConv)
    relu_1 = ReLU()(BN_1)
    conv_1 = Conv2D(
        filters=512,
        padding='same',
        kernel_size=1
    )(relu_1)

    if initial_block:
        x = Conv2D(
            filters=512,
            kernel_size=1,
            padding='same'
        )(input)
        x = BatchNormalization()(x)
    else:
        x = input

    out = Add()([conv_1, x])
    return out


def stage_3_ResBlocK(input: Tensor, kernels=None, groups=None, filters=None, initial_block: bool = False
                     ):
    if kernels is None:
        kernels = [5, 3]
    if filters is None:
        filters = [128, 128]
    if groups is None:
        groups = [4, 1]

    if initial_block:
        # Move the DownSample to the first block of each stage
        input = MaxPooling2D(
            pool_size=3,
            padding='same',
            strides=2
        )(input)

    #
    conv_10 = Conv2D(
        filters=256,
        padding='same',
        kernel_size=1
    )(input)
    BN_0 = BatchNormalization()(conv_10)
    relu_0 = ReLU()(BN_0)

    # PyConv

    GC_0 = GroupConv2D(out_filters=filters[0],
                       kernel_size=kernels[0],
                       groups=groups[0],
                       padding='same'
    )(relu_0)

    GC_1 = GroupConv2D(out_filters=filters[1],
                       kernel_size=kernels[1],
                       groups=groups[1],
                       padding='same'
    )(relu_0)

    GC_PyConv = Concatenate(axis=-1)([GC_0, GC_1])
    BN_1 = BatchNormalization()(GC_PyConv)
    relu_1 = ReLU()(BN_1)
    conv_1 = Conv2D(
        filters=1024,
        padding='same',
        kernel_size=1
    )(relu_1)

    if initial_block:
        x = Conv2D(
            filters=1024,
            kernel_size=1,
            padding='same'
        )(input)
        x = BatchNormalization()(x)
    else:
        x = input

    out = Add()([conv_1, x])
    return out


def stage_4_ResBlocK(input: Tensor, kernels=None, groups=None, filters=None, initial_block: bool = False
                     ):
    if kernels is None:
        kernels = 3
    if filters is None:
        filters = 512
    if groups is None:
        groups = 1

    if initial_block:
        # Move the DownSample to the first block of each stage
        input = MaxPooling2D(
            pool_size=3,
            padding='same',
            strides=2
        )(input)

    #
    conv_10 = Conv2D(
        filters=512,
        padding='same',
        kernel_size=1
    )(input)
    BN_0 = BatchNormalization()(conv_10)
    relu_0 = ReLU()(BN_0)

    # PyConv

    GC_PyConv = Conv2D(
        filters=512,
        kernel_size=3,
        padding='same'
    )(relu_0)
    BN_1 = BatchNormalization()(GC_PyConv)
    relu_1 = ReLU()(BN_1)
    conv_1 = Conv2D(
        filters=2048,
        padding='same',
        kernel_size=1
    )(relu_1)

    if initial_block:
        x = Conv2D(
            filters=2048,
            kernel_size=1,
            padding='same'
        )(input)
        x = BatchNormalization()(x)
    else:
        x = input

    out = Add()([conv_1, x])
    return out


def mkstages(input: Tensor):
    input_ = None
    out = None

    for i in range(3):
        if i is 0:
            input_ = stage_1_ResBlocK(input=input, initial_block=True)
        else:
            input_ = stage_1_ResBlocK(input=input_)
    for i in range(4):
        if i is 0:
            input_ = stage_2_ResBlocK(input=input_, initial_block=True)
        else:
            input_ = stage_2_ResBlocK(input=input_)
    for i in range(6):
        if i is 0:
            input_ = stage_3_ResBlocK(input=input_, initial_block=True)
        else:
            input_ = stage_3_ResBlocK(input=input_)
    for i in range(3):
        if i is 0:
            input_ = stage_4_ResBlocK(input=input_, initial_block=True)
        elif i is 2:
            out = stage_4_ResBlocK(input=input_)
        else:
            input_ = stage_4_ResBlocK(input=input_)
    assert out is not None, 'Something wrong'

    return out


def classifiaction_stage(input:Tensor,classes_num:int):
    GAP = GlobalAveragePooling2D()(input)
    softmax = Dense(name='classification',activation='softmax',units=classes_num)(GAP)
    return softmax


def PyResNet(classes_num:int = 1000):
    i = Input((224, 224, 3))
    x = stage_0(i)
    x = mkstages(x)
    out = classifiaction_stage(x, classes_num)
    model = Model(inputs=i, outputs=out)
    return model

if __name__ == '__main__':
    model = PyResNet()
    plot_model(model=model, to_file="./Model_structure.jpg", show_shapes=True)
    model.summary()
    sys.exit()
