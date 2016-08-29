# -*- coding: utf-8 -*-

"""
@version: ??
@author: 'kw_w'
@license: Apache Licence 
@contact: kw_w@foxmail.com
@site: https://github.com/Trangle
@software: PyCharm
@file: symbol_cnnv2-0.py
@time: 2016/8/18 22:42
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


def get_symbol(num_classes=12):

    # stage 1  224x224
    input_data = mx.symbol.Variable("data")

    # stage 2 112x112
    conv1 = Conv(input_data, num_filter=16, kernel=(3, 3), stride=(2, 2), pad=(1, 1), name='conv1', withBn=True)
    conv2 = Conv(data=conv1, num_filter=16, kernel=(3, 3), pad=(1, 1), name='conv2', withBn=True)
    conv3 = Conv(data=conv2, num_filter=16, kernel=(3, 3), pad=(1, 1), name='conv3', withBn=True)

    # stage 3  56x56
    pool1 = mx.symbol.Pooling(data=conv3, pool_type="max", kernel=(3, 3), pad=(1, 1), stride=(2, 2), name='pool1')
    conv4 = Conv(data=pool1, num_filter=32, name='conv4', withRelu=False)

    conv5 = Conv(pool1, num_filter=32, kernel=(3, 3), pad=(1, 1), name='conv5', withBn=False)

    conv6 = Conv(data=pool1, num_filter=32, kernel=(3, 3), pad=(1, 1), name='conv6', withBn=True)
    conv7 = Conv(data=conv6, num_filter=32, kernel=(3, 3), pad=(1, 1), name='conv7', withBn=True)

    m1 = mx.sym.Concat(*[conv5, conv7], name='merge1')
    conv8 = Conv(data=m1, num_filter=32, name='liner_conv8', withRelu=False)*0.1

    plus1 = conv4 + conv8
    bn1 = mx.sym.BatchNorm(data=plus1, name='plus1_bn')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name='plus1_relu')

    # stage 3  28x28
    pool2 = mx.symbol.Pooling(data=act1, pool_type="max", kernel=(3, 3), pad=(1, 1), stride=(2, 2), name='pool2')
    conv9 = Conv(data=pool2, num_filter=64, name='conv9', withBn=True)

    conv10 = Conv(data=pool2, num_filter=64, name='conv10', withRelu=False)

    conv11 = Conv(data=pool2, num_filter=64, name='conv11')
    conv12 = Conv(data=conv11, num_filter=64, kernel=(1, 3), pad=(0, 1), name='conv12', withBn=True)
    conv13 = Conv(data=conv12, num_filter=64, kernel=(3, 1), pad=(1, 0), name='conv13', withBn=True)

    merge2 = mx.sym.Concat(*[conv10, conv13], name='merge2')

    conv14 = Conv(data=merge2, num_filter=64, name='liner_conv14', withRelu=False)*0.1

    plus2 = conv9 + conv14
    bn = mx.sym.BatchNorm(data=plus2, name='plus2_bn')
    act = mx.sym.Activation(data=bn, act_type='relu', name='plus2_relu')

    # stage 4  14x14
    pool3 = mx.symbol.Pooling(data=act, pool_type="max", kernel=(3, 3), pad=(1, 1), stride=(2, 2), name='pool3')

    conv15 = Conv(data=pool3, num_filter=128, name='conv15')

    conv16 = Conv(data=pool3, num_filter=128, kernel=(3, 3), pad=(1, 1), name='conv16', withBn=True)
    conv17 = Conv(data=conv16, num_filter=128, kernel=(3, 3), pad=(1, 1), name='conv17', withBn=True)

    merge3 = mx.sym.Concat(*[conv15, conv17], name='merge3')

    # stage 5 7x7
    pool4 = mx.symbol.Pooling(data=merge3, pool_type="max", kernel=(3, 3), pad=(1, 1), stride=(2, 2), name='pool4')

    conv18 = Conv(data=pool4, num_filter=256, kernel=(3, 3), pad=(1, 1), name='conv18', withBn=True)

    # stage 6
    pool5 = mx.symbol.Pooling(data=conv18, pool_type="avg", kernel=(7, 7), stride=(1, 1), name='pool5')

    # stage 5  1x1
    flatten = mx.symbol.Flatten(data=pool5, name='flatten')
    dropout1 = mx.symbol.Dropout(data=flatten, p=0.2)

    fc1 = mx.symbol.FullyConnected(data=dropout1, num_hidden=num_classes, name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')


    return softmax


if __name__ == '__main__':
    net = get_symbol(12)
    shape = {'softmax_label': (32, 12), 'data': (32, 3, 224, 224)}
    mx.viz.print_summary(net, shape=shape)
    mx.viz.plot_network(net, title='cnnv2-3', format='pdf', shape=shape).render('cnnv2-3')