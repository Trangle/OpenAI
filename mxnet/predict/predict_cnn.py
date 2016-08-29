import find_mxnet
import submission_cnn
import mxnet as mx
import logging
import argparse
import time
import cv2

parser = argparse.ArgumentParser(description='generate predictions an image classifer on Kaggle Data Science Bowl 1')
parser.add_argument('--batch-size', type=int, default=32,
                    help='the batch size')
parser.add_argument('--data-dir', type=str, default="data224/",
                    help='the input data directory')
parser.add_argument('--gpus', type=str, default='1',
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--model-prefix', type=str,default= "./models/net_cnnv2-4",
                    help='the prefix of the model to load')
parser.add_argument('--num-round', type=int,default= 504,
                    help='the round/epoch to use')
args = parser.parse_args()



# device used
devs = mx.cpu() if args.gpus is None else [
    mx.gpu(int(i)) for i in args.gpus.split(',')]


# Load the pre-trained model
model = mx.model.FeedForward.load(args.model_prefix, args.num_round, ctx=devs, numpy_batch_size=args.batch_size)


# test set data iterator
data_shape = (3, 224, 224)
test = mx.io.ImageRecordIter(
    path_imgrec = args.data_dir + "va.rec",
    mean_r      = 128,
    mean_b      = 128,
    mean_g      = 128,
    scale       = 1.0 / 60,
    rand_crop   = False,
    rand_mirror = False,
    rand_short_crop=False,
    data_shape  = data_shape,
    batch_size  = args.batch_size)

# generate matrix of prediction prob
# X = model._init_iter(test, None, is_train=True)
# for x in X:
#     for i in range(len(x.label)):
#         print(x.label[i].asnumpy().astype('int32'), x.data[i].asnumpy())

# data = test.getdata().asnumpy()
# for i in range(len(data)):
#     img = data[i].transpose((1, 2, 0))
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     cv2.imwrite("result%d.png" % (i), img)

tic=time.time()
[predictions, data, label_o] = model.predict(test, return_data=True)
# for i, x in enumerate(data):
#     img = x.transpose((1, 2, 0))
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     pre = predictions[i].tolist()
#     cv2.imshow(','.join([str(pre.index(max(pre))), str(label_o[i])]), img)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()

print("Time required for prediction", time.time()-tic)


# create submission csv file to submit to kaggle
submission_cnn.gen_sub(predictions,test_lst_path="data/va.lst",submission_path="submission.csv")
