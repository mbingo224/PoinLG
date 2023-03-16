import torch
from torch import nn
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
knn = KNN(k=16, transpose_mode=False)


class DGCNN_Grouper(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(3, 8, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 32),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 64),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 128),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    
    @staticmethod
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

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):
        # x_k: 就是DGCNN中的x_j   x_q：就是 DGCNN 中的 x_i       
        # coor: bs, 3, np, x: bs, c, np
        # 这里coor_q与coor_k实际只是参与knn，获得 coor_q 在 coor_k 的knn近邻点
        k = 16
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = knn(coor_k, coor_q)  # bs k np，原 DGCNN 没有引入查询集与参考集
            assert idx.shape[1] # assert 避免搜寻的邻域点不一致
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base # 区分开不同 batch 的索引
            idx = idx.view(-1) # 展开成一维数组
        # 实际上上述步骤就是get_knn_index()借鉴而来的，获得邻域点的索引
        # 下述代码就是 PoinTr 所引用的获得graph feature的实现函数：get_graph_feature()
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        # 因为这里时修改feature 的原有的值，因此需要使用contiguous，如果右边的表达式是赋给另外的新变量就不需要加contiguous
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous() # B C N K，需要应用contiguous拷贝出一份新的tensor，否则 permute 这些修改不生效
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature # 返回x_q中心点坐标的本身全局特征x_q与考虑邻域的局部特征的 [bs, 2c, np, K(16)]

    def forward(self, x, sample_npoints):

        # x: bs, 3, np

        # bs 3 N(128)   bs C(224)128 N(128)
        coor = x # [bs, 3, 2048]
        f = self.input_trans(x) # bs 3 np -> bs 8 np(2048)
        # [1, 16, 2048, 16], B C N K，两个f：bs 8 np(2048)级联获得
        f = self.get_graph_feature(coor, f, coor, f) # 这里的获得 edge feature的查询集与参考集是相同的
        f = self.layer1(f) # [1, 32, 2048, 16]
        f = f.max(dim=-1, keepdim=False)[0] # point-wise 聚合所有近邻点的特征 [1, 32, 2048] 找到每一行中的最大值

        # fps 采样获得 512个中心点coor_q（[bs, 3, num_group(512)]）及中心点特征f_q[bs, 32, num_group]
        # 这里相比 DGCNN 原网络添加了FPS 来下采样目的是减少点数，减少计算量，
        # 因为后面Transformer 对于处理的序列的长度（点集的数量）是有限的，太长了会导致计算量比较大
        # 实际上这里的 512 的设置来源于 N / k，128 来源于 N / k^2 ，此次采样即是 k = 4
        coor_q, f_q = self.fps_downsample(coor, f, 512)
        f = self.get_graph_feature(coor_q, f_q, coor, f) # [1, 64, 512, 16]
        f = self.layer2(f) # [1, 64, 512, 16] -> # [1, 64, 512, 16]
        f = f.max(dim=-1, keepdim=False)[0] # [bs, C(64), 512]
        coor = coor_q

        f = self.get_graph_feature(coor, f, coor, f) # f: [bs, 128, 512, 16]，EdgeConv获得point-wise feature
        f = self.layer3(f) # f: [bs, 64, 512, 16]
        f = f.max(dim=-1, keepdim=False)[0] # [bs, 64, 512]

        # debug 测试输出不同尺寸的中心点坐标和中心点特征
        # coor_q:[bs, 3, sample_npoints], f_q: [bs, 64, sample_npoints] 
        coor_q, f_q = self.fps_downsample(coor, f, sample_npoints) # 128->sample_npoints
        f = self.get_graph_feature(coor_q, f_q, coor, f) # f: [bs, 128, sample_npoints, k(16)]] EdgeConv获得point-wise feature
        f = self.layer4(f) # f: [bs, 128, sample_npoints, k(16)]
        f = f.max(dim=-1, keepdim=False)[0] # [bs, 128, sample_npoints]
        # problem：这里有别于原 DGCNN 并没有将上述 4 个EdgeConv 模块提取而来的 edge feature 级联

        coor = coor_q # [bs, 3]

        return coor, f # 返回中心点坐标[bs, 3, sample_npoints(128)] 和 中心点特征[bs, 128, sample_npoints(128)]