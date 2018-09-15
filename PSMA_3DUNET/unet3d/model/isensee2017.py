import os
from unet3d.GPU_config import gpuConfig
os.environ["CUDA_VISIBLE_DEVICES"]= gpuConfig['GPU_using']

from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D
from keras.engine import Model
from keras.optimizers import Adam
from keras_contrib.layers.normalization import InstanceNormalization

from .unet import create_convolution_block, concatenate
from ..metrics import weighted_dice_coefficient_loss


create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)


def isensee2017_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=4, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf

    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf


    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """
    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters, conv = True)

        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        # context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        context_output_layer = dense_block_module(in_conv, n_filters = n_level_filters)

        # summation_layer = Add()([in_conv, context_output_layer])
         #level_output_layers.append(summation_layer)
        # current_layer = summation_layer
        level_output_layers.append(context_output_layer)
        print("n_level_filters = ", n_level_filters, "size = current_layer", context_output_layer.shape())
        current_layer = context_output_layer



    segmentation_layers = list()

    current_layer = level_output_layers[-1]
    dense_current_layer = dense_block_module(current_layer, n_filters = n_level_filters)
    concatenation_layer = Add()([current_layer,dense_current_layer])
    localization_output = create_localization_module(concatenation_layer, 16)
    current_layer = localization_output
    segmentation_layers.insert(0, create_convolution_block(current_layer, n_filters=n_labels, kernel=(1,1,1)))




    for level_number in range(depth - 3, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        # Why ?
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        # Why ?
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, create_convolution_block(current_layer, n_filters=n_labels, kernel=(1, 1, 1)))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model


def dense_block_module(input_layer,n_filters,dropout_rate = 0.3, data_format = "channels_first"):
    dense_block_1 = create_convolution_block(input_layer, n_filters, batch_normalization= True)
    dense_block_2 = create_convolution_block(dense_block_1,n_filters, batch_normalization=True)
    concatenation_layer = concatenate([dense_block_1,dense_block_2],axis = 1)
    dense_block_3 = create_localization_module(concatenation_layer, n_filters)
    dropout = SpatialDropout3D(rate = dropout_rate, data_format = data_format)(dense_block_3)
    return  dense_block_3

def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2
