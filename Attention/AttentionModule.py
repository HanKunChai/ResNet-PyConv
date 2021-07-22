from keras.layers import GlobalAveragePooling2D, Dense, Multiply, Reshape, Conv2D, GlobalMaxPooling2D, Add, Activation
from tensorflow import Tensor


def SENet(inputs: Tensor, reduction: int = 16):
    channels = int(inputs.shape[-1])
    assert channels >= reduction, "channels must bigger than reduction"
    gap = GlobalAveragePooling2D()(inputs)
    fc_1 = Dense(units=channels // reduction, activation='relu')(gap)
    fc_2 = Dense(units=channels, activation='hard_sigmoid')(fc_1)
    out = Multiply()([inputs, fc_2])
    return out


def CBAM(input, reduction: int = 16):
    channels = input.shape.as_list()[-1]
    avg_x = GlobalAveragePooling2D()(input)
    avg_x = Reshape((1, 1, channels))(avg_x)
    avg_x = Conv2D(int(channels) // reduction, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   activation='relu')(avg_x)
    avg_x = Conv2D(int(channels), kernel_size=(1, 1), strides=(1, 1), padding='valid')(avg_x)

    max_x = GlobalMaxPooling2D()(input)
    max_x = Reshape((1, 1, channels))(max_x)
    max_x = Conv2D(int(channels) // reduction, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   activation='relu')(max_x)
    max_x = Conv2D(int(channels), kernel_size=(1, 1), strides=(1, 1), padding='valid')(max_x)

    cbam_feature = Add()([avg_x, max_x])

    cbam_feature = Activation('hard_sigmoid')(cbam_feature)

    return Multiply()([input, cbam_feature])
