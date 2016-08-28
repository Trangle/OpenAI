"""

Inception V4, suitable for images with around 299 x 299

Reference:

Szegedy C, Ioffe S, Vanhoucke V. Inception-v4, inception-resnet and the impact of residual connections on learning[J]. arXiv preprint arXiv:1602.07261, 2016.

"""

import find_mxnet
import mxnet as mx


def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix='', withRelu=True, withBn=False):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                              name='%s%s_conv2d' % (name, suffix))
    if withBn:
        conv = mx.sym.BatchNorm(data=conv, name='%s%s_bn' % (name, suffix))
    if withRelu:
        conv = mx.sym.Activation(data=conv, act_type='relu', name='%s%s_relu' % (name, suffix))
    return conv


# Input Shape is 3*224*224 (th)-> 64x56x56
def InceptionResnetStem(data,
                        name):
    c1 = Conv(data=data, num_filter=64, kernel=(7, 7), pad=(3, 3), stride=(2, 2), name=('%s_conv1' % name), withBn=True)
    pool1 = mx.sym.Pooling(data=c1, kernel=(3, 3), pad=(1, 1), stride=(2, 2), pool_type='max', name=('%s_%s_pool1' % ('max', name)))
    c2 = Conv(data=pool1, num_filter=64, kernel=(3, 3), pad=(1, 1), name=('%s_conv2' % name), withRelu=False)
    return c2


def InceptionResnetV2A(data,
                       name,
                       scaleResidual=True):
    init = data
    bn1 = mx.sym.BatchNorm(data=init, name=('%s_a_bn1' % name))
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name=('%s_a_relu1' % name))



    a1 = Conv(data=relu1, num_filter=32, name=('%s_a_1' % name), suffix='_conv', withRelu=False)

    a2 = Conv(data=relu1, num_filter=16, name=('%s_a_2' % name), suffix='_conv_1', withBn=True)
    a2 = Conv(data=a2, num_filter=16, kernel=(3, 3), pad=(1, 1), name=('%s_a_2' % name), suffix='_conv_2', withRelu=False)

    a3 = Conv(data=relu1, num_filter=16, name=('%s_a_3' % name), suffix='_conv_1', withBn=True)
    a3 = Conv(data=a3, num_filter=16, kernel=(3, 3), pad=(1, 1), name=('%s_a_3' % name), suffix='_conv_2', withBn=True)
    a3 = Conv(data=a3, num_filter=16, kernel=(3, 3), pad=(1, 1), name=('%s_a_3' % name), suffix='_conv_3', withRelu=False)

    merge = mx.sym.Concat(*[a1, a2, a3], name=('%s_a_concat1' % name))
    a4 = Conv(data=merge, num_filter=64, name=('%s_a_4' % name), suffix='_conv', withRelu=False)

    if scaleResidual:
        a4 *= 0.1

    out = init + a4

    return out


def InceptionResnetV2B(data,
                       name,
                       scaleResidual=True):
    init = data
    bn1 = mx.sym.BatchNorm(data=init, name=('%s_b_bn1' % name))
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name=('%s_b_relu1' % name))

    b1 = Conv(data=relu1, num_filter=64, name=('%s_b_1' % name), suffix='_conv', withRelu=False)

    b2 = Conv(data=relu1, num_filter=32, name=('%s_b_2' % name), suffix='_conv_1', withBn=True)
    b2 = Conv(data=b2, num_filter=32, kernel=(1, 5), pad=(0, 2), name=('%s_b_2' % name), suffix='_conv_2', withBn=True)
    b2 = Conv(data=b2, num_filter=64, kernel=(5, 1), pad=(2, 0), name=('%s_b_2' % name), suffix='_conv_3', withRelu=False)

    merge = mx.sym.Concat(*[b1, b2], name=('%s_b_concat1' % name))
    b3 = Conv(data=merge, num_filter=128, name=('%s_b_3' % name), suffix='_conv', withRelu=False)

    if scaleResidual:
        b3 *= 0.1

    out = init + b3

    return out


def InceptionResnetV2C(data,
                       name,
                       scaleResidual=True):
    init = data
    bn1 = mx.sym.BatchNorm(data=init, name=('%s_c_bn1' % name))
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name=('%s_c_relu1' % name))

    c1 = Conv(data=relu1, num_filter=128, name=('%s_c_1' % name), suffix='_conv', withRelu=False)

    c2 = Conv(data=relu1, num_filter=64, name=('%s_c_2' % name), suffix='_conv_1', withBn=True)
    c2 = Conv(data=c2, num_filter=64, kernel=(1, 3), pad=(0, 1), name=('%s_c_2' % name), suffix='_conv_2', withBn=True)
    c2 = Conv(data=c2, num_filter=128, kernel=(3, 1), pad=(1, 0), name=('%s_c_2' % name), suffix='_conv_3', withRelu=False)

    merge = mx.sym.Concat(*[c1, c2], name=('%s_c_concat1' % name))
    c3 = Conv(data=merge, num_filter=256, name=('%s_c_3' % name), suffix='_conv', withRelu=False)

    if scaleResidual:
        c3 *= 0.1

    out = init + c3

    return out


def ReductionResnetV2A(data,
                       name):

    bn1 = mx.sym.BatchNorm(data=data, name=('%s_ra_bn1' % name))
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name=('%s_ra_relu1' % name))

    ra1 = mx.sym.Pooling(data=relu1, kernel=(3, 3), pad=(1, 1), stride=(2, 2), pool_type='max', name=('%s_%s_pool1' % ('max', name)))

    ra2 = Conv(data=relu1, num_filter=16, name=('%s_ra_2' % name), suffix='_conv_1', withBn=True)
    ra2 = Conv(data=ra2, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_ra_2' % name), suffix='_conv_2', withRelu=False)

    ra3 = Conv(data=relu1, num_filter=16, name=('%s_ra_3' % name), suffix='_conv_1', withBn=True)
    ra3 = Conv(data=ra3, num_filter=32, kernel=(3, 3), pad=(1, 1), name=('%s_ra_3' % name), suffix='_conv_2', withBn=True)
    ra3 = Conv(data=ra3, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_ra_3' % name), suffix='_conv_3', withRelu=False)

    m = mx.sym.Concat(*[ra1, ra2, ra3], name=('%s_ra_concat1' % name))

    return m


def ReductionResnetV2B(data,
                       name):

    bn1 = mx.sym.BatchNorm(data=data, name=('%s_rb_bn1' % name))
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name=('%s_rb_relu1' % name))

    rb1 = Conv(data=relu1, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_rb_1' % name), suffix='_conv_1', withRelu=False)

    rb2 = Conv(data=relu1, num_filter=32, name=('%s_rb_2' % name), suffix='_conv_1', withBn=True)
    rb2 = Conv(data=rb2, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_rb_2' % name), suffix='_conv_2', withRelu=False)

    rb3 = Conv(data=relu1, num_filter=32, name=('%s_rb_3' % name), suffix='_conv_1', withBn=True)
    rb3 = Conv(data=rb3, num_filter=64, kernel=(3, 3), pad=(1, 1), name=('%s_rb_3' % name), suffix='_conv_2', withBn=True)
    rb3 = Conv(data=rb3, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_rb_3' % name), suffix='_conv_3', withRelu=False)

    m = mx.sym.Concat(*[rb1, rb2, rb3], name=('%s_rb_concat1' % name))

    return m


def circle_in3a(data,
                name,
                scale,
                round):
    in3a = data
    for i in xrange(round):
        in3a = InceptionResnetV2A(in3a,
                                  name + ('_%d' % i),
                                  scaleResidual=scale)
    return in3a


def circle_in2b(data,
                name,
                scale,
                round):
    in2b = data
    for i in xrange(round):
        in2b = InceptionResnetV2B(in2b,
                                  name + ('_%d' % i),
                                  scaleResidual=scale)
    return in2b


def circle_in2c(data,
                name,
                scale,
                round):
    in2c = data
    for i in xrange(round):
        in2c = InceptionResnetV2C(in2c,
                                  name + ('_%d' % i),
                                  scaleResidual=scale)
    return in2c


# create inception-resnet-v2
def get_symbol(num_classes=1000, scale=True):
    # input shape 3*229*229
    data = mx.symbol.Variable(name="data")

    # stage stem

    in_stem = InceptionResnetStem(data,
                                  'stem_stage')

    # stage 2 x Inception Resnet A

    in3a = circle_in3a(in_stem,
                       'in3a',
                       scale,
                       1)

    # stage Reduction Resnet A

    re3a = ReductionResnetV2A(in3a,
                              're3a')

    # stage 2 x Inception Resnet B

    in2b = circle_in2b(re3a,
                       'in2b',
                       scale,
                       2)

    # stage ReductionB

    re3b = ReductionResnetV2B(in2b,
                              're3b')

    # stage 2 x Inception Resnet C

    in2c = circle_in2c(re3b,
                       'in2c',
                       scale,
                       1)

    # stage 5 7x7
    bn1 = mx.sym.BatchNorm(data=in2c, name='g_bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='g_relu1')
    # pool1 = mx.symbol.Pooling(data=relu1, pool_type="max", kernel=(3, 3), pad=(1, 1), stride=(2, 2), name='g_pool1')
    conv1 = Conv(data=relu1, num_filter=256, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name='g_conv1', withBn=True)

    # stage Average Poolingact1
    pool2 = mx.sym.Pooling(data=conv1, kernel=(7, 7), stride=(1, 1), pool_type="avg", name="g_pool2")

    # stage Dropout
    # dropout = mx.sym.Dropout(data=pool2, p=0.2)
    flatten = mx.sym.Flatten(data=pool2, name="flatten")

    # output
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return softmax


if __name__ == '__main__':
    net = get_symbol(12, scale=True)
    shape = {'softmax_label': (32, 12), 'data': (32, 3, 224, 224)}
    mx.viz.print_summary(net, shape=shape)
    mx.viz.plot_network(net, title='incep-res-small', format='pdf', shape=shape).render('incep-res-small')
