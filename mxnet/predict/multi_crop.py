# -*- coding: utf-8 -*-


"""
@version: ??
@author: 'kw_w'
@license: Apache Licence 
@contact: kw_w@foxmail.com
@site: https://github.com/Trangle
@software: PyCharm
@file: multi_crop.py
@time: 2016/9/9 21:20
"""

import cv2
import numpy as np
from skimage import io,transform
import mxnet as mx
import os
import heapq
from itertools import chain

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

def drop_suffix(input):
    ret = []
    for x in input:
        root, file = os.path.split(x)
        file_name, ext = os.path.splitext(file)
        ret.append(file_name)
    return ret

def copyFiles(sourceDir, file, targetDir):
    sourceFile = os.path.join(sourceDir,  file)
    targetFile = os.path.join(targetDir,  file)
    if os.path.isfile(sourceFile):
        if not os.path.exists(targetDir):
            os.makedirs(targetDir)
        if not os.path.exists(targetFile) or(os.path.exists(targetFile) and (os.path.getsize(targetFile) != os.path.getsize(sourceFile))):
                open(targetFile, "wb").write(open(sourceFile, "rb").read())

def get_crop_patch(img, rect):
    return img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

def crop(img_path, short_edges, crop_size, quality):
    img = cv2.imread(img_path)
    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, quality]
    for short_edge in short_edges:
        S = min(img.shape[:2])
        L = max(img.shape[:2])
        scale = 1.0 * short_edge / S
        scale_full = 1.0 * crop_size / L
        new_size = (int(img.shape[1] * scale + 0.5), int(img.shape[0] * scale + 0.5))
        new_size_full = (int(img.shape[1] * scale_full + 0.5), int(img.shape[0] * scale_full + 0.5))

        new_img = cv2.resize(img, new_size)
        new_full_img = cv2.resize(img, new_size_full)

        # ret, buf = cv2.imencode('.jpg', cv2.resize(img, new_size), encode_params)
        # new_img = cv2.imdecode(buf, -1)
        # ret, buf = cv2.imencode('.jpg', cv2.resize(img, new_size_full), encode_params)
        # new_full_img = cv2.imdecode(buf, -1)

        H = new_img.shape[0]
        W = new_img.shape[1]
        patch_left_top = get_crop_patch(new_img, (0, 0, crop_size, crop_size))
        yield patch_left_top
        yield cv2.flip(patch_left_top, 1)
        patch_right_top = get_crop_patch(new_img, (W-crop_size, 0, crop_size, crop_size))
        yield  patch_right_top
        yield cv2.flip(patch_right_top, 1)
        patch_left_bottoom = get_crop_patch(new_img, (0, H-crop_size, crop_size, crop_size))
        yield patch_left_bottoom
        yield cv2.flip(patch_left_bottoom, 1)
        patch_right_bottom = get_crop_patch(new_img, (W-crop_size, H-crop_size, crop_size, crop_size))
        yield patch_right_bottom
        yield cv2.flip(patch_right_bottom, 1)
        patch_center = get_crop_patch(new_img, ((W-crop_size) / 2, (H-crop_size) / 2, crop_size, crop_size))
        yield patch_center
        yield cv2.flip(patch_center, 1)

        patch_full = np.zeros((crop_size, crop_size, 3), np.uint8)
        rect = ((crop_size - new_full_img.shape[1]) / 2, (crop_size - new_full_img.shape[0]) / 2, new_full_img.shape[1], new_full_img.shape[0])
        patch_full[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]=new_full_img.copy()
        yield patch_full
        yield cv2.flip(patch_full, 1)

if __name__ == '__main__':
    pass
    # img_iters = crop(img_path, short_edges, crop_size)
    # prob = None
    # count = 0
    # for img in img_iters:
    #     sample = np.asarray(img)
    #     sample = np.swapaxes(sample, 0, 2)
    #     sample = np.swapaxes(sample, 1, 2)
    #     normed_img = (sample - mean_img) / 50.0
    #     normed_img.resize(1, 3, 224, 224)
    #     count += 1
    #     batch = normed_img
    #     if prob is None:
    #         prob = model.predict(batch)[0]
    #     else:
    #         prob += model.predict(batch)[0]
    # prob /= count
    # pred = np.argsort(prob)[::-1]
    #
    # top1 = pred[0]
    # # copyFiles(source_path, test_img, os.path.join(dest_path, synset[top1]))
    # print(img_path, " Top1: ", synset[top1])
    # top5 = [synset[pred[i]] for i in range(5)]
    # print("Top5: ", top5)
    # print(prob)
    # io.show()