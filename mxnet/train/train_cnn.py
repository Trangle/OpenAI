# -*- coding: utf-8 -*-

"""
@version: ??
@author: 'kw_w'
@license: Apache Licence 
@contact: kw_w@foxmail.com
@site: https://github.com/Trangle
@software: PyCharm
@file: train_cnn.py
@time: 2016/8/18 21:18
"""

from __future__ import print_function
import find_mxnet
import mxnet as mx
import logging
import argparse
import train_model
import time

# don't use -n and -s, which are resevered for the distributed training
parser = argparse.ArgumentParser(description='train an image classifer on Kaggle Data Science Bowl 1')
parser.add_argument('--network', type=str, default='cnnv2-3',
                    help='the cnn to use')
parser.add_argument('--data-dir', type=str, default="data224/",
                    help='the input data directory')
parser.add_argument('--model-prefix', type=str, default="./models/net_cnnv2-3",
                    help='the prefix of the model to load/save')
parser.add_argument('--save-model-prefix', type=str, default="./models/net_cnnv2-3",
                    help='the prefix of the model to save')
parser.add_argument('--lr', type=float, default=.1,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=0.5,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=50,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--clip-gradient', type=float, default=5.,
                    help='clip min/max gradient to prevent extreme value')
parser.add_argument('--num-epochs', type=int, default=2000,
                    help='the number of training epochs')
parser.add_argument('--load-epoch', type=int,
                    help="load the model on an epoch using the model-prefix")
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size')
parser.add_argument('--gpus', type=str, default='1',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
parser.add_argument('--num-examples', type=int, default=4487,
                    help='the number of training examples')
parser.add_argument('--num-classes', type=int, default=12,
                    help='the number of classes')
parser.add_argument('--log-file', type=str, default='log_cnnv2-3',
                    help='the name of log file')
parser.add_argument('--log-dir', type=str, default='Logs',
                    help='directory of the log file')
args = parser.parse_args()

# network
import importlib

net = importlib.import_module('symbol_' + args.network).get_symbol(args.num_classes)


# data
def get_iterator(args, kv):
    data_shape = (3, 224, 224)

    # train data iterator
    train = mx.io.ImageRecordIter(
        path_imgrec=args.data_dir + "tr.rec",
        mean_r=128,
        mean_g=128,
        mean_b=128,
        scale=1.0 / 60,
        max_aspect_ratio=0.35,
        data_shape=data_shape,
        batch_size=args.batch_size,
        rand_crop=True,
        rand_mirror=True,
        rand_short_crop=False
    )

    # validate data iterator
    val = mx.io.ImageRecordIter(
        path_imgrec=args.data_dir + "va.rec",
        mean_r=128,
        mean_b=128,
        mean_g=128,
        scale=1.0 / 60,
        rand_crop=False,
        rand_mirror=False,
        rand_short_crop=False,
        data_shape=data_shape,
        batch_size=args.batch_size
    )

    # A quick work around to prevent mxnet complaining the lack of a softmax_label
    # train.label = mx.io._init_data(train.label, allow_empty=True, default_name='svm_label')
    # val.label = mx.io._init_data(val.label, allow_empty=True, default_name='svm_label')

    return (train, val)


# train
tic = time.time()
train_model.fit(args, net, get_iterator)
print("time elapsed to train model", time.time() - tic)
