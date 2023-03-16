import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
knn = KNN(k=16, transpose_mode=False)

'''
def knn(x, k: int):
    """
    inputs:
    - x: b x npoints1 x num_dims (partical_cloud)
    - k: int (the number of neighbor)

    outputs:
    - idx: int (neighbor_idx)
    """
    # x : (batch_size, feature_dim, num_points)
    # Retrieve nearest neighbor indices

    if torch.cuda.is_available():
        from knn_cuda import KNN

        ref = x.transpose(2, 1).contiguous()  # (batch_size, num_points, feature_dim)
        query = ref
        _, idx = KNN(k=k, transpose_mode=True)(ref, query)

    else:
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx
'''


def get_graph_feature(x, k: int = 20, idx=None):
    """
    inputs:
    - x: b x npoints1 x num_dims (partical_cloud)
    - k: int (the number of neighbor)
    - idx: neighbor_idx

    outputs:
    - feature: b x npoints1 x (num_dims*2)
    """

    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = idx.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature

def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n(2048), 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group) # [bs, npoint(512)]

        combined_x = torch.cat([coor, x], dim=1) # [bs(1), 35, 2048]

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        ) # [1, 35, 512]， 根据 fps_idx([1, 512])去遍历出combined_x：[bs(1), 35, 2048]中的对应列

        new_coor = new_combined_x[:, :3] # 前三行是中心点坐标
        new_x = new_combined_x[:, 3:] # 后32行是中心点特征

        return new_coor, new_x # coor: [bs, 3, num_group] f: [bs, 32, num_group]


class EdgeRes(nn.Module):
    """
    input:
    - inp: b x (num_dims+id) x num_points

    outputs:
    - out: b x num_dims x num_points
    """

    def __init__(self, use_SElayer: bool = False):
        super(EdgeRes, self).__init__()
        self.k = 8
        self.conv1 = torch.nn.Conv2d(8, 64, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.conv3 = torch.nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.conv4 = torch.nn.Conv2d(2176, 512, kernel_size=1, bias=False)
        self.conv5 = torch.nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv6 = torch.nn.Conv2d(512, 128, kernel_size=1, bias=False)

        # debug,        
        self.conv7 = torch.nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.conv8 = torch.nn.Conv2d(256, 128, kernel_size=1, bias=False)


        #self.conv7 = torch.nn.Conv2d(256, 3, kernel_size=1, bias=False)

        self.use_SElayer = use_SElayer
        if use_SElayer:
            self.se1 = SELayer(channel=64)
            self.se2 = SELayer(channel=128)
            self.se4 = SELayer(channel=512)
            self.se5 = SELayer(channel=256)
            self.se6 = SELayer(channel=128)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.bn3 = torch.nn.BatchNorm2d(1024)
        self.bn4 = torch.nn.BatchNorm2d(512)
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.bn7 = torch.nn.BatchNorm2d(3)
        self.bn8 = torch.nn.BatchNorm2d(128)

        self.th = nn.Tanh()

    def forward(self, x):
        coor = x # [bs, 3, 2048]

        npoints = x.size()[2]
        # x: [batch_size, 4, num_points]
        if self.use_SElayer:
            x = get_graph_feature(x, k=self.k)  # [bs, 8, num_points, k]
            x = F.relu(self.se1(self.bn1(self.conv1(x))))  # [bs, 64, num_points, k]
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 64, num_points]
            pointfeat = x  # [batch_size, 64, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 128, num_points, k]
            x = F.relu(self.se2(self.bn2(self.conv2(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        else:
            x = get_graph_feature(x, k=self.k)  # [bs, 8, num_points, k]
            x = F.relu(self.bn1(self.conv1(x)))  # [bs, 64, num_points, k]
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 64, num_points]
            pointfeat = x  # [batch_size, 64, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 128, num_points, k]
            x = F.relu(self.bn2(self.conv2(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]

        x = get_graph_feature(x, k=self.k)  # [bs, 256, num_points, k]
        x = self.bn3(self.conv3(x))  # [batch_size, 1024, num_points, k]
        x = x.max(dim=-1, keepdim=False)[0]  # [bs, 1024, num_points]

        x, _ = torch.max(x, 2)  # [batch_size, 1024]
        x = x.view(-1, 1024)  # [batch_size, 1024]
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)  # [batch_size, 1024, num_points]
        x = torch.cat([x, pointfeat], 1)  # [batch_size, 1088, num_points]

        if self.use_SElayer:
            x = get_graph_feature(x, k=self.k)  # [bs, 2176, num_points, k]
            x = F.relu(self.se4(self.bn4(self.conv4(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 512, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 1024, num_points, k]
            x = F.relu(self.se5(self.bn5(self.conv5(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 256, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 512, num_points, k]
            x = F.relu(self.se6(self.bn6(self.conv6(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        else:
            x = get_graph_feature(x, k=self.k)  # [bs, 2176, num_points, k]
            x = F.relu(self.bn4(self.conv4(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 512, num_points]

            # 1. DGCNN 的方式
            # coor_q, x_q = self.fps_downsample(coor, x, 512)
            # x = get_graph_feature(x_q, k=self.k)  # [bs, 1024, num_points, k]
            # --------*******-------

            # 2. 调整卷积层数和参数的设置的方式
            x = get_graph_feature(x, k=self.k)  # [bs, 512, num_points, k]

            x = F.relu(self.bn5(self.conv5(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 256, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 512, num_points, k]
            x = F.relu(self.bn6(self.conv6(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        
        # 1. DGCNN 的方式
        # coor_q, x_q = self.fps_downsample(coor_q, x, 128)
        # x = get_graph_feature(x_q, k=self.k)  # [bs, 256, num_points, k]
        # --------*******-------

        x = get_graph_feature(x, k=self.k)  # [bs, 512, num_points, k]

        # 2. 调整卷积层数和参数的设置的方式
        coor = self.th(self.conv7(x)) # [bs, 3, num_points, k] 
        coor = coor.max(dim=-1, keepdim=False)[0]  # [bs, 3, num_points]

        f = self.th(self.conv8(x)) # [bs, 128, num_points, k]
        f = f.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        # --------*******-------

        return coor, f


class SELayer(nn.Module):
    """
    input:
        x:(b, c, m, n)

    output:
        out:(b, c, m', n')
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


if __name__=='__main__':
    input = torch.randn(64,2048,3) # bs = 64, N = 2048
    use_SElayer = False
    netG = EdgeRes(use_SElayer)
    coor, f = netG(input)
    print(coor, f)