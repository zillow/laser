import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from .pointnet_utils import PointNetEncoder, feature_transform_reguliarzer


class PointNet(nn.Module):
    def __init__(self, in_chs=9, out_chs=128, feature_transform=False):
        super(PointNet, self).__init__()
        # self.k = num_class
        self.feat = PointNetEncoder(
            global_feat=False, feature_transform=feature_transform, channel=in_chs
        )
        self.conv1 = torch.nn.Conv1d(1088, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 768, 1)
        self.conv3 = torch.nn.Conv1d(768, 512, 1)
        self.conv4 = torch.nn.Conv1d(512, out_chs, 1)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(768)
        self.bn3 = nn.BatchNorm1d(512)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        # x = x.transpose(2,1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        # x = x.view(batchsize, n_pts, self.k)
        return x


class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat, weight):
        loss = F.nll_loss(pred, target, weight=weight)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss
