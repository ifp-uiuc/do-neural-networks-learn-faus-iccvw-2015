import numpy

from anna.layers import layers
import anna.models


class SupervisedModel(anna.models.SupervisedModel):
    batch = 64
    input = layers.Input2DLayer(batch, 1, 96, 96)

    k = float(numpy.random.rand()*1+0.2)
    print '## k = %.3f' % k
    winit1 = k/numpy.sqrt(5*5*1)
    winit2 = k/numpy.sqrt(5*5*64)
    winit3 = k/numpy.sqrt(5*5*128)

    def trec(x):
        return x*(x > 0.0)

    nonlinearity = trec

    conv1 = layers.Conv2DLayer(
        input,
        n_features=64,
        filter_size=5,
        weights_std=winit1,
        pad=(2, 2))
    relu1 = layers.NonlinearityLayer(
        input=conv1,
        nonlinearity=nonlinearity)
    pool1 = layers.Pool2DLayer(
        input=relu1,
        filter_size=2,
        stride=(2, 2))
    conv2 = layers.Conv2DLayer(
        input=pool1,
        n_features=128,
        filter_size=5,
        weights_std=winit2,
        pad=(2, 2))
    relu2 = layers.NonlinearityLayer(
        input=conv2,
        nonlinearity=nonlinearity)
    pool2 = layers.Pool2DLayer(
        input=relu2,
        filter_size=2,
        stride=(2, 2))
    conv3 = layers.Conv2DLayer(
        input=pool2,
        n_features=256,
        filter_size=5,
        weights_std=winit3,
        pad=(2, 2))
    relu3 = layers.NonlinearityLayer(
        input=conv3,
        nonlinearity=nonlinearity)
    pool3 = layers.Pool2DLayer(
        input=relu3,
        filter_size=12,
        stride=(12, 12))

    winitD1 = k/numpy.sqrt(numpy.prod(pool3.get_output_shape()))
    winitD2 = k/numpy.sqrt(300)

    fc4 = layers.DenseLayer(
        input_layer=pool3,
        n_outputs=300,
        weights_std=winitD1,
        init_bias_value=1.0,
        nonlinearity=layers.rectify,
        dropout=0.0)
    output = layers.DenseLayer(
        input_layer=fc4,
        n_outputs=6,
        weights_std=winitD2,
        init_bias_value=0.0,
        nonlinearity=layers.softmax)
