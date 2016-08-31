'''
Deep Residual Learning for Image Recognition, http://arxiv.org/abs/1512.03385
an exmaple of deep residual network for bot

commands & setups:
set following parameters in example/image-classification/train_model.py
    momentum = 0.9,
    wd = 0.0001,
    initializer = mx.init.Xavier(rnd_type="gaussian", factor_type="in", magnitude=2.0)

#first train the network with lr=0.1 for 80 epoch
python example/image-classification/train_bot.py --network bot_resnetV1.0 --num-examples 100000 --lr 0.1 --num-epoch 80
'''

import mxnet as mx
#import find_mxnet

def MyConvFactory(data, num_filter, kernel, stride, pad, type = 'conv'):
    ops = type.split('+')

    out = data
    for op in ops:
        op = op.lower()
        if op == 'conv':
            out = mx.symbol.Convolution(data = out, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)     
        elif op == 'bn':
            out = mx.symbol.BatchNorm(data=out)
        elif op == 'relu':
            out = mx.symbol.Activation(data = out, act_type=op)
        else:
            raise TypeError("Input operation is not support. operation: %s"%op)

    return out

def ResBlock(data, num_filter, dim_match):
    if dim_match == True: # if dimension match
        identity_data = data     
        conv1 = MyConvFactory(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), type = 'conv+bn+relu')
        conv2 = MyConvFactory(data=conv1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), type = 'conv+bn')
        new_data = identity_data + conv2
        return new_data
    else:        
        conv1 = MyConvFactory(data=data, num_filter=num_filter, kernel=(3,3), stride=(2,2), pad=(1,1), type = 'conv+bn+relu')
        conv2 = MyConvFactory(data=conv1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), type = 'conv+bn')

        # adopt project method in the paper when dimension increased
        #project_data = conv_factory(data=data, num_filter=num_filter, kernel=(1,1), stride=(2,2), pad=(0,0), conv_type=1)
        conv_short = MyConvFactory(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), type='conv+bn+relu')
        project_data = mx.symbol.Pooling(data=conv_short, kernel=(2,2), stride=(2,2),pad=(0,0),pool_type='max')
        new_data = project_data + conv2
        return new_data

def ResBlockBottleneck(data, num_filter, dim_match):
    if dim_match == True: # if dimension match
        identity_data = data     
        conv1x1 = MyConvFactory(data=data, num_filter=num_filter/4, kernel=(1,1), stride=(1,1), pad=(0,0), type = 'conv+bn+relu')
        conv3x3 = MyConvFactory(data=conv1x1, num_filter=num_filter/4, kernel=(3,3), stride=(1,1), pad=(1,1), type = 'conv+bn+relu')
        conv1x1 = MyConvFactory(data=conv3x3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), type = 'conv+bn')
        new_data = identity_data + conv1x1
        return new_data
    else:        
        conv1x1 = MyConvFactory(data=data, num_filter=num_filter/4, kernel=(1,1), stride=(2,2), pad=(0,0), type = 'conv+bn+relu')
        conv3x3 = MyConvFactory(data=conv1x1, num_filter=num_filter/4, kernel=(3,3), stride=(1,1), pad=(1,1), type = 'conv+bn+relu')
        conv1x1 = MyConvFactory(data=conv3x3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), type = 'conv+bn')

        # adopt project method in the paper when dimension increased
        #project_data = conv_factory(data=data, num_filter=num_filter, kernel=(1,1), stride=(2,2), pad=(0,0), conv_type=1)
        conv_short = MyConvFactory(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), type='conv+bn+relu')
        project_data = mx.symbol.Pooling(data=conv_short, kernel=(2,2), stride=(2,2),pad=(0,0),pool_type='max')
        new_data = project_data + conv1x1
        return new_data

def ResidualNet(data, nBlocks):
    #fisrt blocks
    for i in range(nBlocks[0]):
        data = ResBlockBottleneck(data=data, num_filter=64, dim_match=True)
    
    #second blocks
    for i in range(nBlocks[1]):
        if i==0:
            data = ResBlockBottleneck(data=data, num_filter=128, dim_match=False)
        else:
            data = ResBlockBottleneck(data=data, num_filter=128, dim_match=True)

    #third blocks
    for i in range(nBlocks[2]):
        if i==0:
            data = ResBlockBottleneck(data=data, num_filter=192, dim_match=False)
        else:
            data = ResBlockBottleneck(data=data, num_filter=192, dim_match=True)

    #fourth blocks
    for i in range(nBlocks[3]):
        if i==0:
            data = ResBlockBottleneck(data=data, num_filter=256, dim_match=False)
        else:
            data = ResBlockBottleneck(data=data, num_filter=256, dim_match=True)
       
    return data

def Stem(data):
    conv1 = MyConvFactory(data=data, num_filter=32, kernel=(7,7), stride=(2,2), pad=(3,3), type='conv+bn+relu')
    pool = mx.symbol.Pooling(data=conv1, kernel=(3,3), stride=(2,2),pad=(1,1),pool_type='max')
    conv2 = MyConvFactory(data=pool, num_filter=64, kernel=(3,3), stride=(1,1), pad=(1,1), type='conv+bn+relu')
    return conv2

def get_symbol(num_classes):
    stem_out=Stem(data=mx.symbol.Variable(name='data'))

    nBlocks = [2,3,4,2]  # number of blocks per stage
    resnet = ResidualNet(stem_out, nBlocks)

    conv_end = MyConvFactory(data=resnet, num_filter=256, kernel=(3,3), stride=(1,1), pad=(1,1), type='conv+bn+relu')
    pool = mx.symbol.Pooling(data=conv_end, kernel=(7,7), pool_type='avg')
    flatten = mx.symbol.Flatten(data=pool, name='flatten')
    fc = mx.symbol.FullyConnected(data=flatten, num_hidden=num_classes,  name='fc1')
    softmax = mx.symbol.SoftmaxOutput(data=fc, name='softmax')
    return softmax

if __name__ == '__main__':
    num_classes = 12
    net = get_symbol(num_classes)
    shape = {'softmax_label': (64, num_classes), 'data': (64, 3, 224, 224)}
    mx.viz.print_summary(net, shape=shape)
    title = 'bot_resnetV1.0'
    dot = mx.viz.plot_network(net, title=title, shape=shape)
    dot.render(title, view=True)


