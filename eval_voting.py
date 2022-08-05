import os
import argparse
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from util.data_util import ModelNet40 as ModelNet40
import numpy as np
from paddle.io import DataLoader
from util.util import IOStream, load_cfg_from_cfg_file, merge_cfg_from_list, load_pretrained_model
import sklearn.metrics as metrics
import random

from model.DGCNN_PAConv import PAConv


def get_parser():
    parser = argparse.ArgumentParser(description='3D Object Classification')
    parser.add_argument('--config', type=str, default='config/dgcnn_paconv_train.yaml', help='config file')
    parser.add_argument('--dataset_root', type=str, default=None, help='dataset root')
    parser.add_argument('--model_path', type=str, default='./output/best_model.pdparams', help='model path')
    parser.add_argument('opts', help='see config/dgcnn_paconv_train.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)

    cfg['manual_seed'] = cfg.get('manual_seed', 0)
    cfg['dataset_root'] = args.dataset_root
    cfg['workers'] = cfg.get('workers', 6)
    cfg['model_path'] = args.model_path
    return cfg


def _init_():
    if not os.path.exists('output'):
        os.makedirs('output')

class PointcloudScale(object):  # input random scaling
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        bsize = pc.shape[0]
        for i in range(bsize):
            xyz = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            scales = paddle.to_tensor(xyz)
            scales = paddle.cast(scales, 'float32')
            # pc[i, :, 0:3] = paddle.mul(pc[i, :, 0:3], scales)
            pc[i, :, 0:3] = pc[i, :, 0:3] * scales
        return pc


def test(args):
    test_loader = DataLoader(ModelNet40(dataset_root=args.dataset_root, partition='test', num_points=args.num_points, pt_norm=False), num_workers=args.workers,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    NUM_PEPEAT = 300
    NUM_VOTE = 10

    model = PAConv(args)

    load_pretrained_model(model, args.model_path)
    model.eval()
    best_acc = 0

    pointscale = PointcloudScale(scale_low=0.8, scale_high=1.18)  # set the range of scaling

    for i in range(NUM_PEPEAT):
        test_true = []
        test_pred = []

        for data, label in test_loader:
            label = label.squeeze()
            pred = 0
            for v in range(NUM_VOTE):
                new_data = paddle.assign(data)
                batch_size = data.shape[0]
                if v > 0:
                    new_data = pointscale(new_data.detach())
                with paddle.no_grad():
                    pred += F.softmax(model(new_data.transpose([0, 2, 1])), axis=1)  # sum 10 preds
            pred /= NUM_VOTE   # avg the preds!
            label = label.reshape([-1])
            pred_choice = pred.argmax(axis=1)
            test_true.append(label.cpu().numpy())
            test_pred.append(pred_choice.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        if test_acc > best_acc:
            best_acc = test_acc
        outstr = 'Voting %d, test acc: %.6f,' % (i, test_acc * 100)
        print(outstr)

    final_outstr = 'Final voting test acc: %.6f,' % (best_acc * 100)
    print(final_outstr)


if __name__ == "__main__":
    args = get_parser()
    _init_()


    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        paddle.seed(args.manual_seed)
        paddle.framework.seed(args.manual_seed)

    test(args)
