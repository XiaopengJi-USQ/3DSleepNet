import keras
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.layers import Layer
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Conv3D
from keras.layers.pooling import (AveragePooling3D, AveragePooling2D)
from keras.layers import BatchNormalization, LSTM
from keras.regularizers import l2
from keras.layers.core import Activation, Dense, Dropout, Lambda, Flatten


def Pseudo_3D_Conv_Block(x, filter_size, bottleneck=False, dropout_rate=None, weight_decay=1e-4):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    if bottleneck:
        inter_channel = filter_size * 4

        x = Conv3D(inter_channel, (1, 1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    x = Conv3D(filter_size, (3, 3, 1), kernel_initializer='he_normal',
               padding='same', use_bias=False)(x)
    x = Conv3D(filter_size, (1, 1, 3), kernel_initializer='he_normal',
               padding='same', use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def Pseudo_2D_Conv_Block(x, filter_size, bottleneck=False, dropout_rate=None, weight_decay=1e-4):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)

    if bottleneck:
        inter_channel = filter_size * 4

        x = tf.keras.layers.Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filter_size, (3, 1), kernel_initializer='he_normal',
               padding='same', use_bias=False)(x)
    x = tf.keras.layers.Conv2D(filter_size, (1, 3), kernel_initializer='he_normal',
               padding='same', use_bias=False)(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x
def __transition_block(x, nb_filter, compression=1.0, weight_decay=1e-4):

    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=concat_axis, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Conv3D(int(nb_filter * compression), (1, 1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)

    return x

def Conv_Block(x, filters=25,kernel_size = (3, 3, 3),strides=(1, 1, 6), activation='relu'):
    x = tf.keras.layers.Conv3D( filters=filters, kernel_size = kernel_size,strides=strides, activation=activation)(x)
    return x


class PDPAtt(Layer):
    def __init__(self, **kwargs):
        super(PDPAtt, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_channels, num_of_freq_bands, data_points,__ = input_shape
        self.U_1 = self.add_weight(name='U_1',
                                   shape=(data_points, num_of_channels),
                                   initializer='uniform',
                                   trainable=True)
        self.U_2 = self.add_weight(name='U_2',
                                   shape=(num_of_freq_bands, data_points),
                                   initializer='uniform',
                                   trainable=True)
        self.b_e = self.add_weight(name='b_e',
                                   shape=(num_of_channels, num_of_freq_bands, data_points),
                                   initializer='uniform',
                                   trainable=True)

        super(PDPAtt, self).build(input_shape)

    def call(self, x):
        tmp_x = Lambda(channel_wise_mean)(x)
        _, num_of_channels, num_of_freq_bands, data_points = tmp_x.shape
        lhs = tf.matmul(tmp_x,  self.U_1)

        product = tf.matmul(lhs, self.U_2)
        S = K.softmax(product + self.b_e)
        S = keras.layers.Reshape([K.int_shape(x)[1], K.int_shape(x)[2],  K.int_shape(x)[3],1])(S)
        rs = keras.layers.multiply([x, S])
        return rs
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], input_shape[2], input_shape[4])




def Spectral_2DCNN(x, input_shape):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = tf.keras.layers.Conv2D(filters=5, kernel_size=3, strides=(1, 1), activation='relu', input_shape = input_shape)(x)
    x = spatial_attention(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)
    x = Pseudo_2D_Conv_Block(x,10, bottleneck=True)
    x = spatial_attention(x)
    x = Pseudo_2D_Conv_Block(x,20, bottleneck=True)
    x = Flatten()(x)
    return x

def spatial_attention(input_tensor):
    tem = input_tensor
    x = Lambda(channel_wise_mean)(input_tensor)
    x = keras.layers.Reshape([K.int_shape(input_tensor)[1], K.int_shape(
        input_tensor)[2],  1])(x)

    nbSpatial = K.int_shape(input_tensor)[1] * K.int_shape(input_tensor)[2]
    if spatial_attention:
        spatial = AveragePooling2D(
            pool_size=[1, 1])(x)
        spatial = keras.layers.Flatten()(spatial)
        spatial = Dense(nbSpatial)(spatial)
        spatial = Activation('sigmoid')(spatial)
        spatial = keras.layers.Reshape(
            [K.int_shape(input_tensor)[1], K.int_shape(input_tensor)[2],  1])(spatial)
        tem = keras.layers.multiply([input_tensor, spatial])
    return tem





def Temporal_3DCNN(x, input_shape):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = tf.keras.layers.Conv3D(filters=25, kernel_size=(3, 3, 5), strides=(1, 1, 3), activation='relu',
                               padding='same',
                               input_shape=input_shape)(x)
    x = PDPAtt()(x)
    x = Pseudo_3D_Conv_Block(x, 50, bottleneck=True)
    x = PDPAtt()(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
    x = Pseudo_3D_Conv_Block(x, 100, bottleneck=True)
    x = PDPAtt()(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
    x = Pseudo_3D_Conv_Block(x, 200, bottleneck=True)
    x = PDPAtt()(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
    x = Flatten()(x)
    return x

def Temporal_Spectral_3DCNN(x, input_shape):
    concat_axis = 1 if K.image_data_format() == 'channels_first' else -1
    x = tf.keras.layers.Conv3D(filters=25, kernel_size=(3, 3, 3), strides=(1, 1, 1), activation='relu',
                               input_shape=input_shape)(x)
    x = PDPAtt()(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
    x = Pseudo_3D_Conv_Block(x, 50, bottleneck=True)
    x = PDPAtt()(x)
    x = Pseudo_3D_Conv_Block(x, 100, bottleneck=True)
    x = PDPAtt()(x)
    x = Pseudo_3D_Conv_Block(x, 200, bottleneck=True)
    x = PDPAtt()(x)
    x = AveragePooling3D((2, 2, 2), strides=(2, 2, 2))(x)
    x = Flatten()(x)
    return x



def channel_wise_mean(x):
    mid = K.mean(x, axis=-1)
    return mid


def Model_3DCNN_3stream(nb_channels, nb_freq_band, temp_length, spe_length):

    tempInput = Input([nb_channels, nb_freq_band, temp_length,1])
    x_t = Temporal_3DCNN(tempInput,(nb_channels, nb_freq_band, temp_length,1))

    speInput = Input([nb_channels, nb_freq_band, spe_length,1])
    x_ts = Temporal_Spectral_3DCNN(speInput,(nb_channels, nb_freq_band, spe_length,1))

    tsInput = Input([nb_channels, nb_freq_band, 1])
    x_s = Spectral_2DCNN(tsInput,(nb_channels, nb_freq_band,1))

    y = keras.layers.concatenate([x_s, x_t, x_ts], axis=-1)
    y = keras.layers.Dense(100)(y)
    y = keras.layers.Reshape([1, K.int_shape(y)[1]])(y)
    y = LSTM(50)(y)

    y = keras.layers.Reshape([K.int_shape(y)[1]])(y)
    y = Flatten()(y)
    y = keras.layers.Dense(5, activation='softmax')(y)
    model = Model([tempInput, speInput, tsInput], y)
    return model

if __name__ == '__main__':
    model = Model_3DCNN_3stream(9, 9, 300, 10)
    print(model.summary())
