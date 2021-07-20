from keras.layers import Conv2D, GlobalAveragePooling2D, Dense, BatchNormalization, ReLU, MaxPooling2D, \
    Concatenate, Add
from tensorflow import Tensor
from keras import Model, Input
from Model.GroupConv2D import GroupConv2D


def stage_0(input_tensor: Tensor):
    conv_0 = Conv2D(
        filters=64,
        kernel_size=3,
        padding='same',
        strides=2
    )(input_tensor)

    conv_1 = Conv2D(
        filters=64,
        kernel_size=3,
        padding='same',
        strides=1
    )(conv_0)

    conv_2 = Conv2D(
        filters=64,
        kernel_size=3,
        padding='same',
        strides=1
    )(conv_1)

    assert isinstance(conv_2, Tensor), "type error"
    return conv_2


def multipleGroupConv(input: Tensor, groups: [int], dilation_rates: [int], filters: int):
    outputs = []
    for i in range(4):
        outputs.append(GroupConv2D(
            out_filters=filters,
            groups=groups[0],
            padding='same',
            kernel_size=3,
            dilation_rate=dilation_rates[0]
        )(input))

    for i in range(3):
        outputs.append(GroupConv2D(
            out_filters=filters,
            groups=groups[1],
            padding='same',
            kernel_size=3,
            dilation_rate=dilation_rates[1]
        )(input))
    for i in range(2):
        outputs.append(GroupConv2D(
            out_filters=filters,
            groups=groups[2],
            padding='same',
            kernel_size=3,
            dilation_rate=dilation_rates[2]
        )(input))
    outputs.append(GroupConv2D(
        kernel_size=3,
        groups=groups[3],
        out_filters=filters,
        dilation_rate=dilation_rates[3]
    )(input))

    out = Concatenate(axis=-1)(outputs)
    return out


def stage_1_block(input: Tensor, initial_block: bool = False):
    dilated_rates = [4, 3, 2, 1]
    groups = [16, 8, 4, 1]
    if initial_block:
        # Move the DownSample to the first block of each stage
        input = MaxPooling2D(
            pool_size=2,
            padding='same',
            strides=2
        )(input)
        add_layer1 = Conv2D(
            filters=256,
            kernel_size=1,
            padding='same'
        )(input)
    else:
        add_layer1 = input

        # Stage 1 block structure:
        # input -> (DownSample ->) conv(1X1) -> BN ->ReLU -> py-dilated_Conv -> BN -> ReLU ->conv(1X1) -> BN
        # input ->(conv(1X1)) -> Add->out

    conv_0 = Conv2D(
        filters=64,
        kernel_size=1,
        padding='same'
    )(input)

    bn_0 = BatchNormalization()(conv_0)
    relu_0 = ReLU()(bn_0)

    # dilated-Py Conv layer

    # GC_0 = GroupConv2D(
    #     kernel_size=3,
    #     out_filters=16,
    #     groups=groups[0],
    #     padding='same',
    #     dilation_rate=dilated_rates[0]
    # )(relu_0)
    #
    # GC_1 = GroupConv2D(
    #     kernel_size=3,
    #     out_filters=16,
    #     groups=groups[1],
    #     padding='same',
    #     dilation_rate=dilated_rates[1]
    # )(relu_0)
    #
    # GC_2 = GroupConv2D(
    #     kernel_size=3,
    #     out_filters=16,
    #     groups=groups[2],
    #     padding='same',
    #     dilation_rate=dilated_rates[2]
    # )(relu_0)
    #
    # GC_3 = GroupConv2D(
    #     kernel_size=3,
    #     out_filters=16,
    #     groups=groups[3],
    #     padding='same',
    #     dilation_rate=dilated_rates[3]
    # )(relu_0)
    #
    # py_dilate_con = Concatenate(axis=-1)([GC_0, GC_1, GC_2, GC_3])
    py_dilate_con = multipleGroupConv(relu_0, groups=groups, dilation_rates=dilated_rates, filters=16)
    bn_1 = BatchNormalization()(py_dilate_con)
    relu_1 = ReLU()(bn_1)

    conv_1 = Conv2D(
        filters=256,
        kernel_size=1,
        padding='same'
    )(relu_1)
    add_layer2 = BatchNormalization()(conv_1)

    output = Add()([add_layer1, add_layer2])

    return output


def stage_2_block(input: Tensor, initial_block: bool = False):
    dilated_rates = [4, 3, 2, 1]
    groups = [16, 8, 4, 1]
    if initial_block:
        # Move the DownSample to the first block of each stage
        input = MaxPooling2D(
            pool_size=2,
            padding='same',
            strides=2
        )(input)
        add_layer1 = Conv2D(
            filters=512,
            kernel_size=1,
            padding='same'
        )(input)
        input = ReLU()(input)
    else:
        add_layer1 = input
        input = BatchNormalization()(input)
        input = ReLU()(input)
        # Stage 2 block structure: input -> (DownSample->)->(BN)->relu-> conv(1X1) -> BN ->ReLU -> py-dilated_Conv ->
        # BN -> ReLU ->conv(1X1) -> BN
        # input ->(conv(1X1)) -> Add->out

    conv_0 = Conv2D(
        filters=128,
        kernel_size=1,
        padding='same'
    )(input)

    bn_0 = BatchNormalization()(conv_0)
    relu_0 = ReLU()(bn_0)

    # dilated-Py Conv layer

    # GC_0 = GroupConv2D(
    #     kernel_size=3,
    #     out_filters=32,
    #     groups=groups[0],
    #     padding='same',
    #     dilation_rate=dilated_rates[0]
    # )(relu_0)
    #
    # GC_1 = GroupConv2D(
    #     kernel_size=3,
    #     out_filters=32,
    #     groups=groups[1],
    #     padding='same',
    #     dilation_rate=dilated_rates[1]
    # )(relu_0)
    #
    # GC_2 = GroupConv2D(
    #     kernel_size=3,
    #     out_filters=32,
    #     groups=groups[2],
    #     padding='same',
    #     dilation_rate=dilated_rates[2]
    # )(relu_0)
    #
    # GC_3 = GroupConv2D(
    #     kernel_size=3,
    #     out_filters=32,
    #     groups=groups[3],
    #     padding='same',
    #     dilation_rate=dilated_rates[3]
    # )(relu_0)
    #
    # py_dilate_con = Concatenate(axis=-1)([GC_0, GC_1, GC_2, GC_3])
    py_dilate_con = multipleGroupConv(relu_0, groups=groups, dilation_rates=dilated_rates, filters=32)

    bn_1 = BatchNormalization()(py_dilate_con)
    relu_1 = ReLU()(bn_1)

    conv_1 = Conv2D(
        filters=512,
        kernel_size=1,
        padding='same'
    )(relu_1)
    add_layer2 = BatchNormalization()(conv_1)

    output = Add()([add_layer1, add_layer2])

    return output


def stage_3_block(input: Tensor, initial_block: bool = False):
    dilated_rates = [4, 3, 2, 1]
    groups = [16, 8, 4, 1]
    if initial_block:
        # Move the DownSample to the first block of each stage
        input = MaxPooling2D(
            pool_size=2,
            padding='same',
            strides=2
        )(input)
        add_layer1 = Conv2D(
            filters=1024,
            kernel_size=1,
            padding='same'
        )(input)
        input = ReLU()(input)
    else:
        add_layer1 = input
        input = BatchNormalization()(input)
        input = ReLU()(input)
        # Stage 2 block structure: input -> (DownSample->)->(BN)->relu-> conv(1X1) -> BN ->ReLU -> py-dilated_Conv ->
        # BN -> ReLU ->conv(1X1) -> BN
        # input ->(conv(1X1)) -> Add->out

    conv_0 = Conv2D(
        filters=256,
        kernel_size=1,
        padding='same'
    )(input)

    bn_0 = BatchNormalization()(conv_0)
    relu_0 = ReLU()(bn_0)

    # dilated-Py Conv layer

    # GC_0 = GroupConv2D(
    #     kernel_size=3,
    #     out_filters=64,
    #     groups=groups[0],
    #     padding='same',
    #     dilation_rate=dilated_rates[0]
    # )(relu_0)
    #
    # GC_1 = GroupConv2D(
    #     kernel_size=3,
    #     out_filters=64,
    #     groups=groups[1],
    #     padding='same',
    #     dilation_rate=dilated_rates[1]
    # )(relu_0)
    #
    # GC_2 = GroupConv2D(
    #     kernel_size=3,
    #     out_filters=64,
    #     groups=groups[2],
    #     padding='same',
    #     dilation_rate=dilated_rates[2]
    # )(relu_0)
    #
    # GC_3 = GroupConv2D(
    #     kernel_size=3,
    #     out_filters=64,
    #     groups=groups[3],
    #     padding='same',
    #     dilation_rate=dilated_rates[3]
    # )(relu_0)
    #
    # py_dilate_con = Concatenate(axis=-1)([GC_0, GC_1, GC_2, GC_3])
    py_dilate_con = multipleGroupConv(relu_0, groups=groups, dilation_rates=dilated_rates, filters=64)

    bn_1 = BatchNormalization()(py_dilate_con)
    relu_1 = ReLU()(bn_1)

    conv_1 = Conv2D(
        filters=1024,
        kernel_size=1,
        padding='same'
    )(relu_1)
    add_layer2 = BatchNormalization()(conv_1)

    output = Add()([add_layer1, add_layer2])

    return output


def stage_4_block(input: Tensor, initial_block: bool = False):
    if initial_block:
        # Move the DownSample to the first block of each stage
        input = MaxPooling2D(
            pool_size=2,
            padding='same',
            strides=2
        )(input)
        add_layer1 = Conv2D(
            filters=2048,
            kernel_size=1,
            padding='same'
        )(input)

    else:
        add_layer1 = input

        # Stage 2 block structure: input -> (DownSample->)->(BN)->relu-> conv(1X1) -> BN ->ReLU -> py-dilated_Conv ->
        # BN -> ReLU ->conv(1X1) -> BN
        # input ->(conv(1X1)) -> Add->out
    bn = BatchNormalization()(input)
    relu = ReLU()(bn)
    conv_0 = Conv2D(
        filters=512,
        kernel_size=1,
        padding='same'
    )(relu)

    bn_0 = BatchNormalization()(conv_0)
    relu_0 = ReLU()(bn_0)

    # resnet Conv layer

    res_conv = Conv2D(
        filters=512,
        kernel_size=3,
        padding='same'
    )(relu_0)

    bn_1 = BatchNormalization()(res_conv)
    relu_1 = ReLU()(bn_1)

    conv_1 = Conv2D(
        filters=2048,
        kernel_size=1,
        padding='same'
    )(relu_1)
    add_layer2 = BatchNormalization()(conv_1)

    add_out = Add()([add_layer1, add_layer2])
    bn_2 = BatchNormalization()(add_out)
    output = ReLU()(bn_2)
    return output


def DilatedPyConvModel(input_shape=(224, 224, 3), classes_num=6):
    input_layer = Input(input_shape)
    Stage_0 = stage_0(input_layer)
    out = None
    input_ = None
    for i in range(3):
        if i is 0:
            input_ = stage_1_block(input=Stage_0, initial_block=True)
        else:
            input_ = stage_1_block(input=input_)
    for i in range(4):
        if i is 0:
            input_ = stage_2_block(input=input_, initial_block=True)
        else:
            input_ = stage_2_block(input=input_)
    for i in range(6):
        if i is 0:
            input_ = stage_3_block(input=input_, initial_block=True)
        else:
            input_ = stage_3_block(input=input_)
    for i in range(3):
        if i is 0:
            input_ = stage_4_block(input=input_, initial_block=True)
        elif i is 2:
            out = stage_4_block(input=input_)
        else:
            input_ = stage_4_block(input=input_)

    GAP = GlobalAveragePooling2D()(out)
    classification = Dense(name='softmax', activation='softmax', units=classes_num)(GAP)

    model = Model(inputs=input_layer, outputs=classification)
    return model


if __name__ == '__main__':
    model = DilatedPyConvModel()
    model.summary()
