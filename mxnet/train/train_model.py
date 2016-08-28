import find_mxnet
import mxnet as mx
import logging
import os
import numpy as np

@mx.optimizer.Optimizer.register
class Nesterov(mx.optimizer.SGD):

    def update(self, index, weight, grad, state):
        """Update the parameters.
        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters
        weight : NDArray
            weight ndarray
        grad : NDArray
            grad ndarray
        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert(isinstance(weight, mx.nd.NDArray))
        assert(isinstance(grad, mx.nd.NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)

        if state:
            mom = state
            mom *= self.momentum
            grad += wd * weight
            mom += grad
            grad += self.momentum * mom
            weight += -lr * grad
        else:
            assert self.momentum == 0.0
            weight += -lr * (grad + self.wd * weight)

    def set_wd_mult(self, args_wd_mult):
        """Set individual weight decay multipler for parameters.
        By default wd multipler is 0 for all params whose name doesn't
        end with _weight, if param_idx2name is provided.

        Parameters
        ----------
        args_wd_mult : dict of string/int to float
            set the wd multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
        """
        self.wd_mult = {}
        for n in self.idx2name.values():
            if not (
                # Test accuracy 0.930288, 0.930889 and 0.929587 in last 3 epochs
                # if decay weight, bias, gamma and beta like Torch
                # https://github.com/torch/optim/blob/master/sgd.lua
                # Test accuracy 0.927784, 0.928385 and 0.927784 in last 3 epochs
                # if only decay weight
                # Test accuracy 0.928586, 0.929487 and 0.927985 in last 3 epochs
                # if only decay weight and gamma
                n.endswith('_weight')
                or n.endswith('_bias')
                or n.endswith('_gamma')
                or n.endswith('_beta')
            ) or 'proj' in n or 'zscore' in n:
                self.wd_mult[n] = 0.0
        if self.sym is not None:
            attr = self.sym.list_attr(recursive=True)
            for k, v in attr.items():
                if k.endswith('_wd_mult'):
                    self.wd_mult[k[:-len('_wd_mult')]] = float(v)
        self.wd_mult.update(args_wd_mult)

    def set_lr_mult(self, args_lr_mult):
        """Set individual learning rate multipler for parameters

        Parameters
        ----------
        args_lr_mult : dict of string/int to float
            set the lr multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
        """
        self.lr_mult = {}
        for n in self.idx2name.values():
            if 'proj' in n or 'zscore' in n:
                self.lr_mult[n] = 0.0
        if self.sym is not None:
            attr = self.sym.list_attr(recursive=True)
            for k, v in attr.items():
                if k.endswith('_lr_mult'):
                    self.lr_mult[k[:-len('_lr_mult')]] = float(v)
        self.lr_mult.update(args_lr_mult)

def fit(args, network, data_loader, batch_end_callback=None):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

    # load model
    model_prefix = args.model_prefix
    if model_prefix is not None:
        model_prefix += "-%d" % (kv.rank)
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params': tmp.arg_params,
                      'aux_params': tmp.aux_params,
                      'begin_epoch': args.load_epoch}
    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    # data
    (train, val) = data_loader(args, kv)

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step=max(int(epoch_size * args.lr_factor_epoch), 1),
            factor=args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
                    args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    model = mx.model.FeedForward(
        ctx=devs,
        # ctx=mx.cpu(0),
        symbol=network,
        num_epoch=args.num_epochs,
        learning_rate=args.lr,
        momentum=0.9,
        wd=0.0001,
        optimizer='Nesterov',
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34),
        **model_args)

    eval_metrics = ['accuracy']
    ## TopKAccuracy only allows top_k > 1
    for top_k in [2, 3, 5]:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=top_k))

    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 50))

    def norm_stat(d):
        return mx.nd.norm(d) / np.sqrt(d.size)

    mon = mx.mon.Monitor(100, norm_stat)
    model.fit(
        X=train,
        eval_data=val,
        eval_metric=eval_metrics,
        # monitor=mon,
        kvstore=kv,
        batch_end_callback=batch_end_callback,
        epoch_end_callback=checkpoint)
