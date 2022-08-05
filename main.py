import argparse
import os

import numpy as np
import paddle
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
    parser.add_argument('--dataset_root', type=str, default=None, help='dataset root')
    parser.add_argument('--log_iters', type=int, default=10, help='dataset root')
    parser.add_argument('--save_dir', type=str, default='./output', help='save dir')
    parser.add_argument('--model_path', type=str, default='./output/best_model.pdparams', help='model path')
    parser.add_argument('opts', help='see config/dgcnn_paconv.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    cfg['dataset_root'] = args.dataset_root
    cfg['log_iters'] = args.log_iters
    cfg['save_dir'] = args.save_dir
    cfg['model_path'] = args.model_path
    cfg['workers'] = cfg.get('workers', 0)
    return cfg


def _init_(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


# weight initialization:
def weight_init(m):
    if isinstance(m, paddle.nn.Linear):
        kaiming_normal_(m.weight)
        if m.bias is not None:
            constant_(m.bias, 0)
    elif isinstance(m, paddle.nn.Conv2D):
        kaiming_normal_(m.weight)
        if m.bias is not None:
            constant_(m.bias, 0)
    elif isinstance(m, paddle.nn.Conv1D):
        kaiming_normal_(m.weight)
        if m.bias is not None:
            constant_(m.bias, 0)
    elif isinstance(m, paddle.nn.BatchNorm2D):
        constant_(m.weight, 1)
        constant_(m.bias, 0)
    elif isinstance(m, paddle.nn.BatchNorm1D):
        constant_(m.weight, 1)
        constant_(m.bias, 0)


def train(args):
    train_loader = DataLoader(
        ModelNet40(dataset_root=args.dataset_root, partition='train', num_points=args.num_points, pt_norm=args.pt_norm),
        num_workers=args.workers, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        ModelNet40(dataset_root=args.dataset_root, partition='test', num_points=args.num_points, pt_norm=False),
        num_workers=args.workers, batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    model = PAConv(args)

    print(str(model))

    model.apply(weight_init)
    lr = CosineAnnealingDecay(learning_rate=args.lr, T_max=args.epochs, eta_min=args.lr / 100)
    opt = Momentum(parameters=model.parameters(), learning_rate=lr, momentum=args.momentum, weight_decay=1e-4)

    criterion = cal_loss

    best_test_acc = 0
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        lr.step()
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for i, (data, label) in enumerate(train_loader):
            data = paddle.transpose(data, [0, 2, 1])
            batch_size = data.shape[0]
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            opt.step()
            opt.clear_gradients()
            preds = logits.argmax(axis=1)
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.numpy())
            train_pred.append(preds.numpy())
            if i % args.log_iters == 0:
                print(f'[Train] epoch:{epoch}\tbatch id:{i}\t lr:{lr.get_lr():<.6f} loss:{loss.item():<.6f}')

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        outstr = '[Train] %d, loss: %.6f, train acc: %.6f, ' % (epoch, train_loss * 1.0 / count, train_acc)
        print(outstr)

        if epoch % 5 == 0:
            do_preciseBN(
                model, train_loader, False,
                min(200, len(train_loader)))
        ####################
        # Test
        ####################
        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        with paddle.no_grad():
            for i, (data, label) in enumerate(test_loader):
                data = paddle.transpose(data, [0, 2, 1])
                batch_size = data.shape[0]
                logits = model(data)
                loss = criterion(logits, label)
                preds = logits.argmax(axis=1)
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.numpy())
                test_pred.append(preds.numpy())
                if i % args.log_iters == 0:
                    print(f'[Test] epoch:{epoch}\tbatch id:{i}\t loss:{loss.item():<.6f}')
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        outstr = '[Test] %d, loss: %.6f, test acc: %.6f,' % (epoch, test_loss * 1.0 / count, test_acc)
        print(outstr)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            print('Max Acc:%.6f' % best_test_acc)
            paddle.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pdparams'))
            paddle.save(opt.state_dict(), os.path.join(args.save_dir, 'best_model.pdopt'))
        paddle.save(model.state_dict(), os.path.join(args.save_dir, f'{epoch}_model.pdparams'))
        paddle.save(opt.state_dict(), os.path.join(args.save_dir, f'{epoch}_model.pdopt'))


def test(args):
    test_loader = DataLoader(
        ModelNet40(dataset_root=args.dataset_root, partition='test', num_points=args.num_points, pt_norm=False),
        batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    model = PAConv(args)

    print(str(model))

    load_pretrained_model(model, args.model_path)
    model.eval()
    test_true = []
    test_pred = []
    for data, label in test_loader:
        label = label.squeeze()
        data = data.transpose([0, 2, 1])
        with paddle.no_grad():
            logits = model(data)
        preds = logits.argmax(axis=1)
        test_true.append(label.numpy())
        test_pred.append(preds.numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f' % (test_acc, avg_per_class_acc)
    print(outstr)


if __name__ == "__main__":
    args = get_parser()
    _init_(args)

    if not args.eval:
        train(args)
    else:
        test(args)
