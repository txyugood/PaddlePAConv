import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def knn(x, k):
    B, _, N = x.shape
    inner = -2 * paddle.matmul(x.transpose([0, 2, 1]), x)
    xx = paddle.sum(x ** 2, axis=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose([0, 2, 1])

    _, idx = pairwise_distance.topk(k=k, axis=-1)  # (batch_size, num_points, k)

    return idx, pairwise_distance


def get_scorenet_input(x, idx, k):
    """(neighbor, neighbor-center)"""
    batch_size = x.shape[0]
    num_points = x.shape[2]
    x = x.reshape([batch_size, -1, num_points])

    idx_base = paddle.arange(0, batch_size).reshape([-1, 1, 1]) * num_points

    idx = idx + idx_base

    idx = idx.reshape([-1])

    _, num_dims, _ = x.shape

    x = paddle.transpose(x, [0, 2, 1])

    # neighbor = x.reshape([batch_size * num_points, -1])[idx.numpy().tolist(), :]
    neighbor = x.reshape([batch_size * num_points, -1])
    neighbor = paddle.gather(neighbor, idx, axis=0)
    neighbor = neighbor.reshape([batch_size, num_points, k, num_dims])

    x = x.reshape([batch_size, num_points, 1, num_dims]).tile([1, 1, k, 1])

    xyz = paddle.concat((neighbor - x, neighbor), axis=3).transpose([0, 3, 1, 2])  # b,6,n,k

    return xyz


def feat_trans_dgcnn(point_input, kernel, m):
    """transforming features using weight matrices"""
    # following get_graph_feature in DGCNN: torch.cat((neighbor - center, neighbor), dim=3)
    B, _, N = point_input.shape  # b, 2cin, n
    point_output = paddle.matmul(point_input.transpose([0, 2, 1]).tile([1, 1, 2]), kernel).reshape(
        [B, N, m, -1])  # b,n,m,cout
    center_output = paddle.matmul(point_input.transpose([0, 2, 1]), kernel[:point_input.shape[1]]).reshape(
        [B, N, m, -1])  # b,n,m,cout
    return point_output, center_output


def feat_trans_pointnet(point_input, kernel, m):
    """transforming features using weight matrices"""
    # no feature concat, following PointNet
    B, _, N = point_input.size()  # b, cin, n
    point_output = paddle.matmul(point_input.transpose([0, 2, 1]), kernel).reshape([B, N, m, -1])  # b,n,m,cout
    return point_output


class ScoreNet(nn.Layer):
    def __init__(self, in_channel, out_channel, hidden_unit=[16], last_bn=False):
        super(ScoreNet, self).__init__()
        self.hidden_unit = hidden_unit
        self.last_bn = last_bn
        self.mlp_convs_hidden = nn.LayerList()
        self.mlp_bns_hidden = nn.LayerList()

        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs_nohidden = nn.Conv2D(in_channel, out_channel, 1, bias_attr=not last_bn)
            if self.last_bn:
                self.mlp_bns_nohidden = nn.BatchNorm2D(out_channel, momentum=0.1)

        else:
            self.mlp_convs_hidden.append(
                nn.Conv2D(in_channel, hidden_unit[0], 1, bias_attr=False))  # from in_channel to first hidden
            self.mlp_bns_hidden.append(nn.BatchNorm2D(hidden_unit[0], momentum=0.1))
            for i in range(1, len(hidden_unit)):  # from 2nd hidden to next hidden to last hidden
                self.mlp_convs_hidden.append(nn.Conv2D(hidden_unit[i - 1], hidden_unit[i], 1, bias_attr=False))
                self.mlp_bns_hidden.append(nn.BatchNorm2D(hidden_unit[i], momentum=0.1))
            self.mlp_convs_hidden.append(
                nn.Conv2D(hidden_unit[-1], out_channel, 1, bias_attr=not last_bn))  # from last hidden to out_channel
            self.mlp_bns_hidden.append(nn.BatchNorm2D(out_channel, momentum=0.1))

    def forward(self, xyz, calc_scores='softmax', bias_attr=0):
        B, _, N, K = xyz.shape
        scores = xyz

        if self.hidden_unit is None or len(self.hidden_unit) == 0:
            if self.last_bn:
                scores = self.mlp_bns_nohidden(self.mlp_convs_nohidden(scores))
            else:
                scores = self.mlp_convs_nohidden(scores)
        else:
            for i, conv in enumerate(self.mlp_convs_hidden):
                if i == len(self.mlp_convs_hidden) - 1:  # if the output layer, no ReLU
                    if self.last_bn:
                        bn = self.mlp_bns_hidden[i]
                        scores = bn(conv(scores))
                    else:
                        scores = conv(scores)
                else:
                    bn = self.mlp_bns_hidden[i]
                    scores = F.relu(bn(conv(scores)))

        if calc_scores == 'softmax':
            scores = F.softmax(scores, axis=1) + bias_attr  # B*m*N*K, where bias may bring larger gradient
        elif calc_scores == 'sigmoid':
            scores = F.sigmoid(scores) + bias_attr  # B*m*N*K
        else:
            raise ValueError('Not Implemented!')

        # scores = scores.permute(0, 2, 3, 1)  # B*N*K*m
        scores = paddle.transpose(scores, [0, 2, 3, 1])

        return scores
