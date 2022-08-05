import argparse
import os

import numpy as np
import paddle
import h5py
import sklearn.metrics as metrics
from paddle.io import DataLoader
from paddle.optimizer import Momentum
from paddle.optimizer.lr import CosineAnnealingDecay

from model.DGCNN_PAConv import PAConv
from model.param_init import kaiming_normal_, constant_
from precise_bn import do_preciseBN
from util.data_util import ModelNet40 as ModelNet40
from util.util import cal_loss, load_cfg_from_cfg_file, merge_cfg_from_list, load_pretrained_model


def get_parser():
    parser = argparse.ArgumentParser(description='3D Object Classification')
    parser.add_argument('--config', type=str, default='config/dgcnn_paconv.yaml', help='config file')
    parser.add_argument('--input_file', type=str, default=None, help='dataset root')
    parser.add_argument('--model_path', type=str, default=None, help='model path')
    parser.add_argument('opts', help='see config/dgcnn_paconv.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    cfg['input_file'] = args.input_file
    cfg['model_path'] = args.model_path
    return cfg


def predict(args):

    with open("data/modelnet40_ply_hdf5_2048/shape_names.txt", 'r') as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]

    model = PAConv(args)
    load_pretrained_model(model, args.model_path)
    model.eval()
    f = h5py.File(args.input_file, mode='r')
    data = f['data'][:].astype('float32')
    f.close()
    data = data[:,:args.num_points,:]
    data = data.transpose([0, 2, 1])
    data = paddle.to_tensor(data)
    with paddle.no_grad():
        logits = model(data)
    preds = logits.argmax(axis=1).numpy()
    print(f"The input points is class {labels[preds[0]]}")


if __name__ == "__main__":
    args = get_parser()

    predict(args)
