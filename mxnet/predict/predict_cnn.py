import find_mxnet
import submission_cnn
import mxnet as mx
import logging
import argparse
import time
import cv2
import numpy as np
import os
import heapq
import pandas as pd
from itertools import chain

model_prefixes = ['H:/data/bot_animal/models/net_incep-res-3', 'H:/data/bot_animal/models/net_incep-res-2']
model_rounds = [[184], [271]]


def read_files(file_pth, header):
    # ret = {}
    img_lst = pd.read_csv(file_pth, sep="\t", header=header)
    img_lst = img_lst.values.tolist()[:2]
    # map(lambda x: ret.setdefault(x[0], int(x[1])), img_lst)
    return img_lst


class TopkHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []

    def Push(self, elem):
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                heapq.heapreplace(self.data, elem)

    def TopK(self):
        return [x for x in reversed([heapq.heappop(self.data) for _ in xrange(len(self.data))])]


def topk(input, k):
    th = TopkHeap(k)
    for i in input:
        th.Push(i)
    return th.TopK()


class BtmkHeap(object):
    def __init__(self, k):
        self.k = k
        self.data = []

    def Push(self, elem):
        # Reverse elem to convert to max-heap
        elem = -elem
        # Using heap algorighem
        if len(self.data) < self.k:
            heapq.heappush(self.data, elem)
        else:
            topk_small = self.data[0]
            if elem > topk_small:
                heapq.heapreplace(self.data, elem)

    def BtmK(self):
        return sorted([-x for x in self.data])


def Btmk(input, k):
    th = BtmkHeap(k)
    for i in input:
        th.Push(i)
    return th.BtmK()


def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions an image classifer on Kesci Bot Animal')
    parser.add_argument('--multi-crop', type=bool, default=False,
                        help='Whether use multiple crop')
    parser.add_argument('--multi-length', type=int, default=5,
                        help='if use multiple crop, then this is enabled.')
    parser.add_argument('--crop-size', type=int, default=12,
                        help='basic crop size in multiple crop.')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='the batch size')
    parser.add_argument('--mean-data', type=str, default='mean.bin',
                        help='the binary mean file name')
    parser.add_argument('--data-dir', type=str, default="H:/data/bot_animal/data/mean/",
                        help='the input data directory')
    parser.add_argument('--test-dir', type=str, default="H:/data/bot_animal/data/vali/",
                        help='the input data directory')
    parser.add_argument('--dest-dir', type=str, default="H:/data/bot_animal/predict_data/",
                        help='the output data directory')
    parser.add_argument('--gpus', type=str, default='1',
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--model-prefix', type=str, default="H:/data/bot_animal/models/net_incep-res-3",
                        help='the prefix of the model to load')
    parser.add_argument('--num-round', type=int, default=184,
                        help='the round/epoch to use')
    parser.add_argument('--test-dataset', type=str, default="vali.rec",
                        help="test dataset name")
    parser.add_argument('--test-list', type=str, default="vali.lst",
                        help="test list file name")
    return parser.parse_args()


def iter_data(args, data_shape):
    data = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(args.test_dir, args.test_dataset),
        # mean_r=128,
        # mean_b=128,
        # mean_g=128,
        mean_img=os.path.join(args.test_dir, args.mean_data),
        scale=1.0 / 50,
        rand_crop=False,
        rand_mirror=False,
        # rand_short_crop=False,
        data_shape=data_shape,
        batch_size=args.batch_size
    )
    return data


if __name__ == '__main__':
    args = parse_args()
    args.multi_crop = True
    args.multi_length = 6
    args.crop_size = 12
    args.batch_size = 100
    multi_size = args.multi_length * args.crop_size
    ret = read_files(os.path.join(args.test_dir, args.test_list), header=None)
    data_shape = (3, 224, 224)
    # device used
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # # Load the pre-trained model
    # models = []
    # prediction = None
    # # data =[]
    # label_o = []
    #
    # for index, model_prefix in enumerate(model_prefixes):
    #     for m in model_rounds[index]:
    #         models.append(mx.model.FeedForward.load(model_prefix, m, ctx=devs, numpy_batch_size=args.batch_size))
    #
    #
    # tic=time.time()
    # count = 0
    # for index, model in enumerate(models):
    #     test = iter_data(args, data_shape)
    #     tic_tmp = time.time()
    #     [tmp_pre, _, tmp_label] = model.predict(test, return_data=True)
    #     label_o = tmp_label
    #     count += 1
    #     if prediction is None:
    #         prediction = tmp_pre
    #     else:
    #         try:
    #             prediction += tmp_pre
    #         except Exception as ee:
    #             print('numpy array plus error: ', ee)
    #             exit(0)
    #     print("Time required for prediction %s" % index, time.time()-tic_tmp)
    # try:
    #     prediction /= count
    #     prediction = prediction.tolist()
    #     print("Time required for all prediction", time.time() - tic)
    #
    #     new_pred = []
    #     correct = np.zeros(2)
    #     count = 0
    #     for j, pred in enumerate(prediction):
    #         x = topk(pred, 2)
    #         y=['%.6f' % v for v in x]
    #         index = [pred.index(i) for i in x]
    #         curr_label = int(label_o[j])
    #         if index[0] == curr_label or index[1] == curr_label:
    #             correct[1] += 1
    #             if index[0] == curr_label:
    #                 correct[0] += 1
    #         count += 1
    #         new_pred.append(list(chain(*zip(index, y))))
    #     print('top1: %s, top2: %s' % (1.0 * correct[0] / count, 1.0 * correct[1] / count))
    #
    #     # create submission csv file to submit to kesci_bot_animal
    #     submission_cnn.gen_sub(new_pred, test_lst_path=os.path.join(args.test_dir, args.test_list), submission_path="submission.txt")
    # except Exception as e:
    #     print('numpy array divide error: ', e)
    #     exit(0)

    model = mx.model.FeedForward.load(args.model_prefix, args.num_round, ctx=devs, numpy_batch_size=args.batch_size)

    # test set data iterator

    test = mx.io.ImageRecordIter(
        path_imgrec=os.path.join(args.test_dir, args.test_dataset),
        # mean_r      = 128,
        # mean_b      = 128,
        # mean_g      = 128,
        mean_img=os.path.join(args.test_dir, args.mean_data),
        scale=1.0 / 50,
        rand_crop=False,
        rand_mirror=False,
        # rand_short_crop=False,
        data_shape=data_shape,
        batch_size=args.batch_size
    )

    # generate matrix of prediction prob
    # X = model._init_iter(test, None, is_train=True)
    # for x in X:
    #     for i in range(len(x.label)):
    #         print(x.label[i].asnumpy().astype('int32'), x.data[i].asnumpy())
    #
    # data = test.getdata().asnumpy()
    # for i in range(len(data)):
    #     img = data[i].transpose((1, 2, 0))
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     cv2.imwrite("result%d.png" % (i), img)

    tic = time.time()
    # [predictions, _, label_o] = model.predict(test, return_data=True)
    predictions = model.predict(test)

    correct = np.zeros(2)
    count = 0
    pre = {}
    for i, x in enumerate(ret):
        prob = predictions[i]
        tmp_key = str(x[0]) + ' ' + str(x[1])
        if pre.has_key(tmp_key):
            pre[tmp_key] += prob
        else:
            pre[tmp_key] = prob
    for d, x in pre.items():
        count += 1
        pred = np.argsort(x)[::-1]
        curr_label = int(d.split(' ')[1])
        if pred[0] == curr_label or pred[1] == curr_label:
            correct[1] += 1
            if pred[0] == curr_label:
                correct[0] += 1
    print('top1: %s, top2: %s' % (1.0 * correct[0] / count, 1.0 * correct[1] / count))
    print("Time required for all predictions", time.time() - tic)
    # for i, x in enumerate(data):
    #     img = x.transpose((1, 2, 0))
    #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     pre = predictions[i].tolist()
    #     cv2.imshow(','.join([str(pre.index(max(pre))), str(label_o[i])]), img)
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
