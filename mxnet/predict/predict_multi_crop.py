# -*- coding: utf-8 -*-


"""
@version: ??
@author: 'kw_w'
@license: Apache Licence 
@contact: kw_w@foxmail.com
@site: https://github.com/Trangle
@software: PyCharm
@file: predict_multi_crop.py
@time: 2016/9/10 0:50
"""
import multi_crop as mcrp
import submission_cnn
import argparse
import numpy as np
from skimage import io,transform
import mxnet as mx
import os
from itertools import chain

def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions an image classifer on Kesci Bot Animal')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='the batch size')
    parser.add_argument('--mean-data', type=str, default='mean.bin',
                        help='the binary mean file name')
    parser.add_argument('--data-dir', type=str, default="H:/data/bot_animal/data/mean/",
                        help='the input data directory')
    parser.add_argument('--test-dir', type=str, default="H:/data/bot_animal/data/Testset2/",
                        help='the input data directory')
    parser.add_argument('--dest-dir', type=str, default="H:/data/bot_animal/predict_data/",
                        help='the output data directory')
    parser.add_argument('--gpus', type=str, default='0',
                        help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--model-prefix', type=str, default= "H:/data/bot_animal/models/net_incep-res-2",
                        help='the prefix of the model to load')
    parser.add_argument('--num-round', type=int,default= 106,
                        help='the round/epoch to use')
    parser.add_argument('--test-dataset', type=str, default="test.rec",
                        help="test dataset name")
    parser.add_argument('--test-list', type=str, default="test.lst",
                        help="test list file name")
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    short_edges = [224, 256, 320, 384, 480, 640]
    # short_edges = [224, 256, 288]
    crop_size = 224
    mean_img = mx.nd.load(os.path.join(args.data_dir, args.mean_data)).values()[0].asnumpy()
    # img_path = 'H:/data/bot_animal/data/Testset2/0a492d3afe114b1cb26668ceb0d28017.jpg'

    # device used
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    model = mx.model.FeedForward.load(args.model_prefix,args.num_round, ctx=devs, numpy_batch_size=1)
    synset = "00_guinea_pig,01_squirrel,02_sikadeer,03_fox,04_dog,05_wolf,06_cat,07_chipmuck,08_giraffe,09_reindeer,10_hyena,11_weasel".split(
        ',')
    test_imgs = os.listdir(args.test_dir)
    img_name = mcrp.drop_suffix(test_imgs)

    new_pred = []
    for i, test_img in enumerate(test_imgs):
        img_ori = io.imread(os.path.join(args.test_dir, test_img))
        io.imshow(img_ori)

        img_iters = mcrp.crop(os.path.join(args.test_dir, test_img), short_edges, crop_size, 90)
        prob = None
        count = 0
        for img in img_iters:
            sample = np.asarray(img)
            sample = np.swapaxes(sample, 0, 2)
            sample = np.swapaxes(sample, 1, 2)
            # print('imageLen: %s, imageName: %s' % (len(sample), img_name[i]))
            normed_img = (sample - mean_img) / 50.0
            # normed_img = sample.copy()
            normed_img.resize(1, 3, 224, 224)
            count += 1
            batch = normed_img
            if prob is None:
                prob = model.predict(batch)[0]
            else:
                prob += model.predict(batch)[0]

        prob /= count
        prob_list = prob.tolist()
        x = mcrp.topk(prob_list, 2)
        y = ['%.6f' % v for v in x]
        index = [prob_list.index(j) for j in x]
        new_pred.append(list(chain(*zip(index, y))))
        pred = np.argsort(prob)[::-1]
        top1 = pred[0]
        # copyFiles(source_path, test_img, os.path.join(dest_path, synset[top1]))
        print(img_name[i], " Top1: ", synset[top1])
        top5 = [synset[pred[i]] for i in range(5)]
        print("Top5: ", top5)
        print(prob)
        io.show()
    # create submission csv file to submit to kesci_bot_animal
    # submission_cnn.gen_sub(new_pred, test_lst_path=os.path.join(args.test_dir, args.test_list),
    #                        submission_path="submission.txt")
    print('total %d images predicted.' % len(new_pred))