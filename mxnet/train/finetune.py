# -*- coding: utf-8 -*-

from __future__ import print_function

import find_mxnet
import submission_dsb
import mxnet as mx
import logging
import argparse
import time
import scipy.io as sio

def parse_args():
    parser = argparse.ArgumentParser(description='generate predictions an image classifer')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--data-dir', type=str, default="data224/",
                        help='the input data directory')
    parser.add_argument('--gpus', type=str, default='0,1',
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--model-prefix', type=str,default= "./models/sample_net",
                        help='the prefix of the model to load')
    parser.add_argument('--num-round', type=int,default= 1,
                        help='the round/epoch to use')
    parser.add_argument('--data-shape', type=int, default=224,
                        help='set image\'s shape')
    return  parser.parse_args()


def get_feature_symbol(model, top_layer=None):
    """Get feature symbol from a model
    .. note::
        If top_layer is not present, it will return the second last layer symbol
    Parameters
    ----------
    model: mx.model.FeedForward
        Model will be used to extract feature symbol
    top_layer: str, option
        Name of top_layer will be used
    Returns
    -------
    internals[top_layer]: mx.symbol.Symbol
        Feature symbol
    """
    internals = model.symbol.get_internals()
    tmp = internals.list_outputs()[::-1]
    outputs = [name for name in tmp if name.endswith("output")]
    if top_layer != None and type(top_layer) != str:
        error_msg = "top_layer must be a string in following candidates:\n %s" % "\n".join(outputs)
        raise TypeError(error_msg)
    if top_layer == None:
        assert len(outputs) > 3
        top_layer = outputs[2]
    else:
        if top_layer not in outputs:
            error_msg = "%s not exists in symbol. Possible choice:\n%s" \
                    % (top_layer, "\n".join(outputs))
            raise ValueError(error_msg)
    return internals[top_layer]


if __name__ == "__main__":

    args = parse_args()

    # device used
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # Load the pre-trained model
    model = mx.model.FeedForward.load(args.model_prefix, args.num_round, ctx=devs, numpy_batch_size=args.batch_size)
    # test set data iterator
    data_shape = (3, args.data_shape, args.data_shape)
    test = mx.io.ImageRecordIter(
        path_imgrec=args.data_dir + "va.rec",
        mean_r=128,
        mean_b=128,
        mean_g=128,
        scale=1.0 / 60,
        rand_crop=False,
        rand_mirror=False,
        data_shape=data_shape,
        batch_size=args.batch_size)

    # generate matrix of prediction prob
    tic = time.time()
    predictions = model.predict(test)
    print("Time required for prediction", time.time() - tic)

    internals = model.symbol.get_internals()
    # 记住要提取特征的那一层的名字。 我这是 flatten 。
    fea_symbol = internals["flatten_output"]
    feature_extractor = mx.model.FeedForward(ctx=mx.gpu(), symbol=fea_symbol, numpy_batch_size=1,
                                             arg_params=model.arg_params, aux_params=model.aux_params,
                                             allow_extra_params=True)
    [val_feature, valdata, vallabel] = feature_extractor.predict(test, return_data=True)

    sio.savemat('tmp/flatten_1.mat', {'val_feature': val_feature})