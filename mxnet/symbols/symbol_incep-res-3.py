"""

Inception V4, suitable for images with around 299 x 299

Reference:

Szegedy C, Ioffe S, Vanhoucke V. Inception-v4, inception-resnet and the impact of residual connections on learning[J]. arXiv preprint arXiv:1602.07261, 2016.

"""

import find_mxnet
import mxnet as mx


def Conv(data, num_filter, bn_momentum=0.9, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix='', withRelu=True, withBn=True):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True,
                              name='%s%s_conv2d' % (name, suffix))
    if withBn:
        conv = mx.sym.BatchNorm(data=conv, momentum=bn_momentum, fix_gamma=False, eps=2e-5, name='%s%s_bn' % (name, suffix))
        # conv = mx.sym.BatchNorm(data=conv, name='%s%s_bn' % (name, suffix))
    if withRelu:
        conv = mx.sym.Activation(data=conv, act_type='relu', name='%s%s_relu' % (name, suffix))
    return conv


# Input Shape is 3*224*224 (th)-> 64x56x56
def InceptionResnetStem(data,
                        name):
    c1 = Conv(data=data, num_filter=32, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_conv1' % name))
    c2 = Conv(data=c1, num_filter=64, kernel=(3, 3), pad=(1, 1), name=('%s_conv2' % name))
    pool1 = mx.sym.Pooling(data=c2, kernel=(3, 3), pad=(1, 1), stride=(2, 2), pool_type='max', name=('%s_%s_pool1' % ('max', name)))
    c3 = Conv(data=pool1, num_filter=64, kernel=(3, 3), pad=(1, 1), name=('%s_conv3' % name))
    c4 = Conv(data=c3, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_conv4' % name), withBn=False, withRelu=False)
    return c4


def InceptionResnetV2A(data,
                       name,
                       bn_momentum=0.9,
                       scaleResidual=True):

    bn1 = mx.symbol.BatchNorm(name=('%s_bn1' % name), data=data, fix_gamma=False, momentum=bn_momentum, eps=2e-5)
    ac1 = mx.symbol.Activation(name=('%s_relu1' % name), data=bn1, act_type='relu')

    init = data

    a1 = Conv(data=ac1, num_filter=32, name=('%s_a_1' % name), suffix='_conv')

    a2 = Conv(data=ac1, num_filter=16, name=('%s_a_2' % name), suffix='_conv_1')
    a2 = Conv(data=a2, num_filter=32, kernel=(3, 3), pad=(1, 1), name=('%s_a_2' % name), suffix='_conv_2')

    a3 = Conv(data=ac1, num_filter=16, name=('%s_a_3' % name), suffix='_conv_1')
    a3 = Conv(data=a3, num_filter=32, kernel=(3, 3), pad=(1, 1), name=('%s_a_3' % name), suffix='_conv_2')
    a3 = Conv(data=a3, num_filter=64, kernel=(3, 3), pad=(1, 1), name=('%s_a_3' % name), suffix='_conv_3')

    merge = mx.sym.Concat(*[a1, a2, a3], name=('%s_a_concat1' % name))
    a4 = Conv(data=merge, num_filter=128, name=('%s_a_4' % name), suffix='_conv', withRelu=False, withBn=False)

    if scaleResidual:
        a4 *= 0.4

    out = init + a4

    return out


def InceptionResnetV2B(data,
                       name,
                       bn_momentum=0.9,
                       scaleResidual=True):

    bn1 = mx.symbol.BatchNorm(name=('%s_bn1' % name), data=data, fix_gamma=False, momentum=bn_momentum, eps=2e-5)
    ac1 = mx.symbol.Activation(name=('%s_relu1' % name), data=bn1, act_type='relu')

    init = data

    b1 = Conv(data=ac1, num_filter=64, name=('%s_b_1' % name), suffix='_conv')

    b2 = Conv(data=ac1, num_filter=32, name=('%s_b_2' % name), suffix='_conv_1')
    b2 = Conv(data=b2, num_filter=48, kernel=(1, 7), pad=(0, 3), name=('%s_b_2' % name), suffix='_conv_2')
    b2 = Conv(data=b2, num_filter=64, kernel=(7, 1), pad=(3, 0), name=('%s_b_2' % name), suffix='_conv_3')

    merge = mx.sym.Concat(*[b1, b2], name=('%s_b_concat1' % name))
    b3 = Conv(data=merge, num_filter=256, name=('%s_b_3' % name), suffix='_conv', withRelu=False, withBn=False)

    if scaleResidual:
        b3 *= 0.4

    out = init + b3

    return out


def InceptionResnetV2C(data,
                       name,
                       bn_momentum=0.9,
                       scaleResidual=True):
    bn1 = mx.symbol.BatchNorm(name=('%s_bn1' % name), data=data, fix_gamma=False, momentum=bn_momentum, eps=2e-5)
    ac1 = mx.symbol.Activation(name=('%s_relu1' % name), data=bn1, act_type='relu')

    init = data

    c1 = Conv(data=ac1, num_filter=128, name=('%s_c_1' % name), suffix='_conv')

    c2 = Conv(data=ac1, num_filter=64, name=('%s_c_2' % name), suffix='_conv_1')
    c2 = Conv(data=c2, num_filter=96, kernel=(1, 3), pad=(0, 1), name=('%s_c_2' % name), suffix='_conv_2')
    c2 = Conv(data=c2, num_filter=128, kernel=(3, 1), pad=(1, 0), name=('%s_c_2' % name), suffix='_conv_3')

    merge = mx.sym.Concat(*[c1, c2], name=('%s_c_concat1' % name))
    c3 = Conv(data=merge, num_filter=608, name=('%s_c_3' % name), suffix='_conv', withRelu=False, withBn=False)

    if scaleResidual:
        c3 *= 0.4

    out = init + c3

    return out


def ReductionResnetV2A(data,
                       name,
                       bn_momentum=0.9):

    bn1 = mx.symbol.BatchNorm(name=('%s_bn1' % name), data=data, fix_gamma=False, momentum=bn_momentum, eps=2e-5)
    ac1 = mx.symbol.Activation(name=('%s_relu1' % name), data=bn1, act_type='relu')

    ra1 = mx.sym.Pooling(data=ac1, kernel=(3, 3), pad=(1, 1), stride=(2, 2), pool_type='max', name=('%s_%s_pool1' % ('max', name)))

    ra2 = Conv(data=ac1, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_ra_2' % name), suffix='_conv_2')

    ra3 = Conv(data=ac1, num_filter=32, name=('%s_ra_3' % name), suffix='_conv_1')
    ra3 = Conv(data=ra3, num_filter=48, kernel=(3, 3), pad=(1, 1), name=('%s_ra_3' % name), suffix='_conv_2')
    ra3 = Conv(data=ra3, num_filter=64, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_ra_3' % name), suffix='_conv_3')

    m = mx.sym.Concat(*[ra1, ra2, ra3], name=('%s_ra_concat1' % name))
    m = Conv(data=m, num_filter=256, name=('%s_ra_sc' % name), suffix='_conv', withRelu=False, withBn=False)

    return m


def ReductionResnetV2B(data,
                       name,
                       bn_momentum = 0.9):
    bn1 = mx.symbol.BatchNorm(name=('%s_bn1' % name), data=data, fix_gamma=False, momentum=bn_momentum, eps=2e-5)
    ac1 = mx.symbol.Activation(name=('%s_relu1' % name), data=bn1, act_type='relu')

    rb1 = mx.sym.Pooling(data=data, kernel=(3, 3), pad=(1, 1), stride=(2, 2), pool_type='max', name=('%s_%s_pool1' % ('max', name)))

    rb2 = Conv(data=data, num_filter=64, name=('%s_rb_2' % name), suffix='_conv_1')
    rb2 = Conv(data=rb2, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_rb_2' % name), suffix='_conv_2')

    rb3 = Conv(data=data, num_filter=64, name=('%s_rb_3' % name), suffix='_conv_1')
    rb3 = Conv(data=rb3, num_filter=96, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_rb_3' % name),
               suffix='_conv_2')

    rb4 = Conv(data=data, num_filter=64, name=('%s_rb_4' % name), suffix='_conv_1')
    rb4 = Conv(data=rb4, num_filter=96, kernel=(3, 3), pad=(1, 1), name=('%s_rb_4' % name), suffix='_conv_2')
    rb4 = Conv(data=rb4, num_filter=128, kernel=(3, 3), pad=(1, 1), stride=(2, 2), name=('%s_rb_4' % name), suffix='_conv_3')

    m = mx.sym.Concat(*[rb1, rb2, rb3, rb4], name=('%s_rb_concat1' % name))
    m = Conv(data=m, num_filter=608, name=('%s_rb_sc' % name), suffix='_conv', withRelu=False, withBn=False)

    return m


def circle_in3a(data,
                name,
                scale,
                bn_momentum,
                round):
    in3a = data
    for i in xrange(round):
        in3a = InceptionResnetV2A(in3a,
                                  name + ('_%d' % i),
                                  bn_momentum=bn_momentum,
                                  scaleResidual=scale)
    return in3a


def circle_in2b(data,
                name,
                scale,
                bn_momentum,
                round):
    in2b = data
    for i in xrange(round):
        in2b = InceptionResnetV2B(in2b,
                                  name + ('_%d' % i),
                                  bn_momentum=bn_momentum,
                                  scaleResidual=scale)
    return in2b


def circle_in2c(data,
                name,
                scale,
                bn_momentum,
                round):
    in2c = data
    for i in xrange(round):
        in2c = InceptionResnetV2C(in2c,
                                  name + ('_%d' % i),
                                  bn_momentum=bn_momentum,
                                  scaleResidual=scale)
    return in2c


# create inception-resnet-v2
def get_symbol(num_classes=1000, bn_momentum=0.9, scale=True):
    # input shape 3*229*229
    data = mx.symbol.Variable(name="data")
    zscore = mx.symbol.BatchNorm(name='bn_data', data=data, fix_gamma=True, momentum=bn_momentum, eps=2e-5)
    # stage stem

    in_stem = InceptionResnetStem(zscore,
                                  'stem_stage')
    # round = [3, 1, 0]
    round = [3, 6, 2]
    # stage 2 x Inception Resnet A

    in3a = circle_in3a(in_stem,
                       'in3a',
                       scale,
                       bn_momentum,
                       round[0])

    # stage Reduction Resnet A

    re3a = ReductionResnetV2A(in3a,
                              're3a',
                              bn_momentum)

    # stage 2 x Inception Resnet B

    in2b = circle_in2b(re3a,
                       'in2b',
                       scale,
                       bn_momentum,
                       round[1])

    # stage ReductionB

    re3b = ReductionResnetV2B(in2b,
                              're3b',
                              bn_momentum)

    # stage 2 x Inception Resnet C

    in2c = circle_in2c(re3b,
                       'in2c',
                       scale,
                       bn_momentum,
                       round[2])

    bn1 = mx.symbol.BatchNorm(name='bn1', data=in2c, fix_gamma=False, momentum=bn_momentum, eps=2e-5)
    ac1 = mx.symbol.Activation(name='relu1', data=bn1, act_type='relu')

    # stage Average Poolingact1
    pool = mx.sym.Pooling(data=ac1, kernel=(7, 7), stride=(1, 1), pool_type="avg", name="g_pool", global_pool=True)

    # stage Dropout
    dropout = mx.sym.Dropout(data=pool, p=0.2)
    flatten = mx.sym.Flatten(data=dropout, name="flatten")

    # output
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes, name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return softmax


if __name__ == '__main__':
    net = get_symbol(12, scale=True)
    shape = {'softmax_label': (32, 12), 'data': (32, 3, 224, 224)}
    mx.viz.print_summary(net, shape=shape)
    mx.viz.plot_network(net, title='incep-res-3', save_format='pdf', shape=shape).render('./files/incep-res-3')
