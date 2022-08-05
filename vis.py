import os
import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np


def draw(x, y, z, name, file_dir, color=None):
    """
    绘制单个样本的三维点图
    """
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'tan', 'orangered', 'lightgreen', 'coral', 'aqua', 'gold',
              'plum', 'khaki', 'cyan', 'crimson', 'lawngreen', 'thistle', 'skyblue', 'lightblue', 'moccasin',
              'pink', 'lightpink', 'fuchsia', 'chocolate', 'tomato', 'orchid', 'grey', 'plum', 'peru', 'purple',
              'teal', 'sienna', 'turquoise', 'violet', 'wheat', 'yellowgreen', 'deeppink', 'azure', 'ivory',
              'brown']
    if color is None:
        for i in range(len(x)):
            ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
            save_name = name + '-{}.png'.format(i)
            save_name = os.path.join(file_dir, save_name)
            ax.scatter(x[i], y[i], z[i], c=colors[i % len(colors)])
            ax.set_zlabel('Z')  # 坐标轴
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
            plt.draw()
            plt.savefig(save_name)
            plt.clf()

            # plt.show()
    else:
        for i in range(len(x)):
            ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
            save_name = name + '-{}-{}.png'.format(i, color[i])
            save_name = os.path.join(file_dir, save_name)
            ax.scatter(x[i], y[i], z[i], c=colors[color[i]])
            ax.set_zlabel('Z')  # 坐标轴
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
            plt.draw()
            plt.savefig(save_name)
            plt.clf()
            # plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='3D Object Classification')
    parser.add_argument('--input_file', type=str, default='', help='input file')
    parser.add_argument('--save_dir', type=str, default='./vis_output', help='input file')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    f = h5py.File(args.input_file, mode='r')
    n_points = f['data'][:].astype('float32')
    n_points = np.transpose(n_points, [0, 2, 1])
    file_dir = './'
    save_name_prefix = 'vis'
    draw(n_points[:, 0, :], n_points[:, 1, :], n_points[:, 2, :], save_name_prefix, args.save_dir)

