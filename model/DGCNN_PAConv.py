"""
Embed PAConv into DGCNN
"""

import paddle.nn as nn
import paddle
import paddle.nn.functional as F
from util.PAConv_util import get_scorenet_input, knn, feat_trans_dgcnn, ScoreNet
from model.param_init import kaiming_normal_
# from custom_ops import assign_score_withk as assemble_dgcnn
from assign_score_gpu import assign_score_withk as assemble_dgcnn

class PAConv(nn.Layer):
    def __init__(self, args):
        super(PAConv, self).__init__()
        self.args = args
        self.k = args.get('k_neighbors', 20)
        self.calc_scores = args.get('calc_scores', 'softmax')

        self.m1, self.m2, self.m3, self.m4 = args.get('num_matrices', [8, 8, 8, 8])
        self.scorenet1 = ScoreNet(6, self.m1, hidden_unit=[16])
        self.scorenet2 = ScoreNet(6, self.m2, hidden_unit=[16])
        self.scorenet3 = ScoreNet(6, self.m3, hidden_unit=[16])
        self.scorenet4 = ScoreNet(6, self.m4, hidden_unit=[16])

        i1 = 3  # channel dim of input_1st
        o1 = i2 = 64  # channel dim of output_1st and input_2nd
        o2 = i3 = 64  # channel dim of output_2st and input_3rd
        o3 = i4 = 128  # channel dim of output_3rd and input_4th
        o4 = 256  # channel dim of output_4th

        params = paddle.zeros(shape=[self.m1, i1 * 2, o1], dtype='float32')
        kaiming_normal_(params, nonlinearity='relu')
        params = paddle.transpose(params, [1, 0, 2]).reshape([i1 * 2, self.m1 * o1])
        matrice1 = paddle.create_parameter(shape=[i1 * 2, self.m1 * o1], dtype='float32',
                                           default_initializer=nn.initializer.Assign(params))
        self.add_parameter('matrice1', matrice1)

        params = paddle.zeros(shape=[self.m2, i2 * 2, o2], dtype='float32')
        kaiming_normal_(params, nonlinearity='relu')
        params = paddle.transpose(params, [1, 0, 2]).reshape([i2 * 2, self.m2 * o2])
        matrice2 = paddle.create_parameter(shape=[i2 * 2, self.m2 * o2], dtype='float32',
                                           default_initializer=nn.initializer.Assign(params))
        self.add_parameter('matrice2', matrice2)

        params = paddle.create_parameter(shape=[self.m3, i3 * 2, o3], dtype='float32')
        kaiming_normal_(params, nonlinearity='relu')
        params = paddle.transpose(params, [1, 0, 2]).reshape([i3 * 2, self.m3 * o3])
        matrice3 = paddle.create_parameter(shape=[i3 * 2, self.m3 * o3], dtype='float32',
                                           default_initializer=nn.initializer.Assign(params))
        self.add_parameter('matrice3', matrice3)

        params = paddle.create_parameter(shape=[self.m4, i4 * 2, o4], dtype='float32')
        kaiming_normal_(params, nonlinearity='relu')
        params = paddle.transpose(params, [1, 0, 2]).reshape([i4 * 2, self.m4 * o4])
        matrice4 = paddle.create_parameter(shape=[i4 * 2, self.m4 * o4], dtype='float32',
                                           default_initializer=nn.initializer.Assign(params))
        self.add_parameter('matrice4', matrice4)


        self.bn1 = nn.BatchNorm1D(o1)
        self.bn2 = nn.BatchNorm1D(o2)
        self.bn3 = nn.BatchNorm1D(o3)
        self.bn4 = nn.BatchNorm1D(o4)
        self.bn5 = nn.BatchNorm1D(1024)
        self.conv5 = nn.Sequential(nn.Conv1D(512, 1024, kernel_size=1, bias_attr=False),
                                   self.bn5)

        self.linear1 = nn.Linear(2048, 512, bias_attr=False)
        self.bn11 = nn.BatchNorm1D(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256, bias_attr=False)
        self.bn22 = nn.BatchNorm1D(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, 40)

    def forward(self, x, label=None, criterion=None):
        B, C, N = x.shape
        idx, _ = knn(x, k=self.k)  # different with DGCNN, the knn search is only in 3D space
        xyz = get_scorenet_input(x, idx=idx, k=self.k)  # ScoreNet input: 3D coord difference concat with coord: b,6,n,k

        ##################
        # replace all the DGCNN-EdgeConv with PAConv:
        """CUDA implementation of PAConv: (presented in the supplementary material of the paper)"""
        """feature transformation:"""
        point1, center1 = feat_trans_dgcnn(point_input=x, kernel=self.matrice1, m=self.m1)  # b,n,m1,o1
        score1 = self.scorenet1(xyz, calc_scores=self.calc_scores, bias_attr=0.5)
        """assemble with scores:"""
        point1 = assemble_dgcnn(scores=score1, points=point1, centers=center1, knn_idx=idx)  # b,o1,n
        point1 = F.relu(self.bn1(point1))

        point2, center2 = feat_trans_dgcnn(point_input=point1, kernel=self.matrice2, m=self.m2)
        score2 = self.scorenet2(xyz, calc_scores=self.calc_scores, bias_attr=0.5)
        point2 = assemble_dgcnn(scores=score2, points=point2, centers=center2, knn_idx=idx)
        point2 = F.relu(self.bn2(point2))

        point3, center3 = feat_trans_dgcnn(point_input=point2, kernel=self.matrice3, m=self.m3)
        score3 = self.scorenet3(xyz, calc_scores=self.calc_scores, bias_attr=0.5)
        point3 = assemble_dgcnn(scores=score3, points=point3, centers=center3, knn_idx=idx)
        point3 = F.relu(self.bn3(point3))

        point4, center4 = feat_trans_dgcnn(point_input=point3, kernel=self.matrice4, m=self.m4)
        score4 = self.scorenet4(xyz, calc_scores=self.calc_scores, bias_attr=0.5)
        point4 = assemble_dgcnn(scores=score4, points=point4, centers=center4, knn_idx=idx)
        point4 = F.relu(self.bn4(point4))
        ##################

        point = paddle.concat((point1, point2, point3, point4), axis=1)
        point = F.relu(self.conv5(point))
        point11 = F.adaptive_max_pool1d(point, 1).reshape([B, -1])
        point22 = F.adaptive_avg_pool1d(point, 1).reshape([B, -1])
        point = paddle.concat((point11, point22), 1)

        point = F.relu(self.bn11(self.linear1(point)))
        point = self.dp1(point)
        point = F.relu(self.bn22(self.linear2(point)))
        point = self.dp2(point)
        point = self.linear3(point)

        if criterion is not None:
            return point, criterion(point, label)  # return output and loss
        else:
            return point
