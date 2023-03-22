import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN

# 目前先使用自定义的 KNN 计算，这里的K才是实际控制邻域点数量的参数，前面的k调整没有用
#knn_index = KNN(k=8, transpose_mode=True)


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
        #print(k)
        # 原函数调用出现问题
        _, idx = KNN(k=k, transpose_mode=True)(ref, query)
        #_, idx = knn_index(ref, query)

    else:
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx



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

class SpareNetEncode(nn.Module):
    """
    input
    - point_cloud: b x num_dims x npoints1

    output
    - feture:  one x feature_size
    """

    def __init__(
        self,
        bottleneck_size=4096,
        use_SElayer=False,
        encode="Residualnet",
        hide_size=2048,
        output_size=4096,
        k = 8
    ):
        super(SpareNetEncode, self).__init__()
        #print_log(msg, logger=logger)
        #print(encode)
        if encode == "Residualnet":
            self.feat_extractor = EdgeConvResFeat(
                use_SElayer=use_SElayer, k = k, output_size=output_size, hide_size=hide_size
            )
        else:
            self.feat_extractor = PointNetfeat(
                global_feat=True, use_SElayer=use_SElayer, hide_size=hide_size
            )
        self.linear_1 = nn.Linear(bottleneck_size , bottleneck_size // 2)
        self.linear_2 = nn.Linear(bottleneck_size // 2, bottleneck_size // 4)
        self.linear_3 = nn.Linear(bottleneck_size // 4, bottleneck_size // 8)
        self.linear_4 = nn.Linear(bottleneck_size // 8, 128 * 3)

        #----------****实验6****----------
        #-----***方法1：第一种方式求取 f ***-------
        # self.fc_1 = nn.Linear(bottleneck_size // 2,128 * 512)
        # self.conv1_1 = torch.nn.Conv1d(512,512,1)#torch.nn.Conv1d(256,256,1) !
        # self.conv1_2 = torch.nn.Conv1d(512,256,1)
        # self.conv1_3 = torch.nn.Conv1d(256,128,1)
        #-----***方法1：第一种方式求取 f ***-------

        #-----***方法2：另外一种方式求取 f ***-------
        # self.fc_1 = nn.Linear(bottleneck_size // 8,128 * 512)
        # self.conv1_1 = torch.nn.Conv1d(512,512,1)#torch.nn.Conv1d(256,256,1) !
        # self.conv1_2 = torch.nn.Conv1d(512,256,1)
        # self.conv1_3 = torch.nn.Conv1d(256,128,1)
        #-----***方法2：另外一种方式求取 f ***-------
        #----------****实验6****----------

        #----------****实验7****----------
        self.fc_1 = nn.Linear(bottleneck_size // 8,128 * 512)
        self.conv1_1 = torch.nn.Conv1d(512,512,1)#torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512,256,1)
        self.conv1_3 = torch.nn.Conv1d(256,128,1)
        #----------****实验7****----------

        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        #----------****实验5****----------
        # coor, f = self.feat_extractor(x)
        # return coor, f
        #----------****实验5****----------

        #----------****实验6****----------
        '''
        ------****NOTE: 在实验6中, 我们都是选择在特征维度为512去线性层变换到目标点云, 但这不是绝对的, 也可以从C=1024
        就开始进行特征变换到 128 * 3 需要进行实验去确定这个值 可参考 MRNet _netG class 中pc2_xyz的生成******-------
        '''
        #-----***方法1：第一种方式求取 f ***-------
        # x = self.feat_extractor(x)
        # x = self.relu(self.linear_1(x)) # [bs, 2048]
        # # 这里计算 f 的原理是: 在高维特征就获得 f 使得 f 包含的特征信息更丰富，因为 x 在之前已经级联了各层次的特征了
        # f = x # [bs, 2048] 实际上这里 f = x 放在哪个位置还需要再实验测试哪个位置效果好一些
        # x = self.relu(self.linear_2(x)) # [bs, 1024]
        # x = self.relu(self.linear_3(x)) # [bs, 512]
        # '''按照 pc2_xyz 的生成, 这一步 self.linear_4 应该设计成: self.linear_4 = nn.Linear(512,64*128)'''
        # x = self.relu(self.linear_4(x)) # [bs, 128 * 3] 
        #-----***方法1：第一种方式求取 f ***-------

        #-----***方法2：另外一种方式求取 f ***-------
        # x = self.feat_extractor(x)
        # x = self.relu(self.linear_1(x))      
        # x = self.relu(self.linear_2(x))
        # x = self.relu(self.linear_3(x)) # [bs, 512]
        # # 这里计算 f 的原理是：保证特征 f 的获取和中心点坐标 coor 的层次是一致的，推测会有利于 Transformer 中的 x + pos
        # f = x # [bs, 512] 实际上这里 f = x 放在哪个位置还需要再实验测试哪个位置效果好一些
        '''按照 pc2_xyz 的生成, 这一步 self.linear_4 应该设计成: self.linear_4 = nn.Linear(512,64*128)'''
        # x = self.relu(self.linear_4(x)) # [bs, 128 * 3]
        #-----***方法2：另外一种方式求取 f ***-------

        # [bs, 3, 128]，这里是为了满足Transformer encorder的输入 shape 要求，因此转置
        # coor = x.reshape(-1, 128, 3).transpose(1, 2).contiguous() 

        # #-----***方法1：第一种方式求取 f ***-------
        # f = self.relu(self.fc_1(f)) # [bs, 2048]->[bs, 512 * 128]
        # f = f.reshape(-1,512,128) # [bs, 512, 128]
        # f = self.relu(self.conv1_1(f))
        # f = self.relu(self.conv1_2(f))
        # f = self.relu(self.conv1_3(f)) # [bs, 128, 128] [B, C, N]
        #-----***方法1：第一种方式求取 f ***-------
        
        #-----***方法2：另外一种方式求取 f ***-------
        # f = self.relu(self.fc_1(f)) # [bs, 512]->[bs, 512 * 128]
        # f = f.reshape(-1,512,128) # [bs, 512, 128] # [bs, 512]->[bs, 512 * 128]
        # f = self.relu(self.conv1_1(f))
        # f = self.relu(self.conv1_2(f))
        # f = self.relu(self.conv1_3(f)) # [bs, 128, 128] [B, C, N]
        #-----***方法2：另外一种方式求取 f ***-------
        
        #return coor, f # coor:[bs, 3, 128]，f: [bs, 128, 128] [B, C, N]

        #----------****实验6****----------


        #----------****实验7****----------
        coor, f = self.feat_extractor(x)

        f = self.relu(self.linear_1(f)) # [bs, 2048]
        f = self.relu(self.linear_2(f)) # [bs, 1024]
        f = self.relu(self.linear_3(f)) # [bs, 512]

         #-----***方法1：第一种方式求取 f ***-------
        f = self.relu(self.fc_1(f)) # [bs, 512]->[bs, 512 * 128]
        f = f.reshape(-1,512,128) # [bs, 512, 128]
        f = self.relu(self.conv1_1(f)) # nn.Conv1d 
        f = self.relu(self.conv1_2(f))
        f = self.relu(self.conv1_3(f)) # [bs, 128, 128] [B, C, N]
        #-----***方法1：第一种方式求取 f ***-------

        return coor, f
        #----------****实验7****----------


class EdgeConvResFeat(nn.Module):
    """
    input
    - point_cloud: b x num_dims x npoints1

    output
    - feture:  b x feature_size
    """

    def __init__(
        self,
        num_point: int = 16382,
        use_SElayer: bool = False,
        k: int = 8,
        hide_size: int = 2048,
        output_size: int = 4096,
    ):
        super(EdgeConvResFeat, self).__init__()
        self.use_SElayer = use_SElayer
        self.k = k
        self.hide_size = hide_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(6, self.hide_size // 16, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(
            self.hide_size // 8, self.hide_size // 16, kernel_size=1, bias=False
        )
        self.conv3 = nn.Conv2d(
            self.hide_size // 8, self.hide_size // 8, kernel_size=1, bias=False
        )
        self.conv4 = nn.Conv2d(
            self.hide_size // 4, self.hide_size // 4, kernel_size=1, bias=False
        )
        self.conv5 = nn.Conv1d(
            self.hide_size // 2, self.output_size // 2, kernel_size=1, bias=False
        )

       
        # self.input_proj = nn.Sequential(
        #     nn.Conv1d(512, 384, 1),
        #     nn.BatchNorm1d(384),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(384, 384, 1)
        # )

        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2)
        self.relu5 = nn.LeakyReLU(negative_slope=0.2)
        #self.relu6 = nn.LeakyReLU(negative_slope=0.2)
        

        if use_SElayer:
            self.se1 = SELayer(channel=self.hide_size // 16)
            self.se2 = SELayer(channel=self.hide_size // 16)
            self.se3 = SELayer(channel=self.hide_size // 8)
            self.se4 = SELayer(channel=self.hide_size // 4)

        self.bn1 = nn.BatchNorm2d(self.hide_size // 16)
        self.bn2 = nn.BatchNorm2d(self.hide_size // 16)
        self.bn3 = nn.BatchNorm2d(self.hide_size // 8)
        self.bn4 = nn.BatchNorm2d(self.hide_size // 4)
        self.bn5 = nn.BatchNorm1d(self.output_size // 2)

        self.resconv1 = nn.Conv1d(
            self.hide_size // 16, self.hide_size // 16, kernel_size=1, bias=False
        )
        self.resconv2 = nn.Conv1d(
            self.hide_size // 16, self.hide_size // 8, kernel_size=1, bias=False
        )
        self.resconv3 = nn.Conv1d(
            self.hide_size // 8, self.hide_size // 4, kernel_size=1, bias=False
        )

    def forward(self, x):
        # x : [bs, 3, num_points]
        batch_size = x.size(0)
        coor = x # [bs, 3, 2048]
        #(self.k)
        if self.use_SElayer:
            x = get_graph_feature(x, k=self.k)
            x = self.relu1(self.se1(self.bn1(self.conv1(x))))
            x1 = x.max(dim=-1, keepdim=False)[0]

            x2_res = self.resconv1(x1)
            x = get_graph_feature(x1, k=self.k)
            x = self.relu2(self.se2(self.bn2(self.conv2(x))))
            x2 = x.max(dim=-1, keepdim=False)[0]
            x2 = x2 + x2_res

            x3_res = self.resconv2(x2)
            x = get_graph_feature(x2, k=self.k)
            x = self.relu3(self.se3(self.bn3(self.conv3(x))))
            x3 = x.max(dim=-1, keepdim=False)[0]
            x3 = x3 + x3_res

            x4_res = self.resconv3(x3)
            x = get_graph_feature(x3, k=self.k)
            x = self.relu4(self.se4(self.bn4(self.conv4(x))))
        else:
            x = get_graph_feature(x, k=self.k) # [bs, 6, num_points, k]
            x = self.relu1(self.bn1(self.conv1(x))) # [bs, 128, num_points, k]
            x1 = x.max(dim=-1, keepdim=False)[0] # [bs, 128, num_points]

            x2_res = self.resconv1(x1) # [bs, 128, num_points] 使用一维卷积增加非线性化
            x = get_graph_feature(x1, k=self.k) # [bs, 256, num_points, k]
            x = self.relu2(self.bn2(self.conv2(x))) # [bs, 128, num_points, k]
            x2 = x.max(dim=-1, keepdim=False)[0] # [bs, 128, num_points]
            x2 = x2 + x2_res # [bs, 128, num_points] 加上 x1 经过一维卷积，增加非线性度同时获得点云中感兴趣的特征

            x3_res = self.resconv2(x2)
            x = get_graph_feature(x2, k=self.k) # [bs, 256, num_points, k]
            x = self.relu3(self.bn3(self.conv3(x))) # [bs, 256, num_points, k]
            x3 = x.max(dim=-1, keepdim=False)[0] # [bs, 256, num_points]
            x3 = x3 + x3_res # [bs, 256, num_points]

            x4_res = self.resconv3(x3) 
            x = get_graph_feature(x3, k=self.k) # [bs, 512, num_points, k]
            x = self.relu4(self.bn4(self.conv4(x))) # [bs, 512, num_points, k]
        x4 = x.max(dim=-1, keepdim=False)[0] # [bs, 512, num_points]
        x4 = x4 + x4_res # [bs, 512, num_points]

        x = torch.cat((x1, x2, x3, x4), dim=1) # [bs, 256/1024, num_points]，PointTr 中DGCNN并没有级联前面提取初、中、高层次特征
        
        #------****Trick：是否可以将这里的conv5修改成 Transformer中的self.input_proj，这样就不必再Transformer里去提升维数
        x = self.relu5(self.bn5(self.conv5(x))) # [bs, 128/2048, num_points] 一维卷积
        # x = self.relu5(self.bn5(self.input_proj(x))) # [bs, 384, num_points] 一维卷积


        #----------****实验5****----------
        # coor, f = fps_downsample(coor, x, 512)
        # coor, f = fps_downsample(coor, f, 256)
        # coor, f = fps_downsample(coor, f, 128)

        # return coor, f
        #----------****实验5****----------

        #----------****实验6****----------
        # 对 x:[bs, 4096] 使用 MLP->reshape 去重构中心点云coor 和 f

        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [bs, 128/2048]，npoints 维度归1，特征被聚合，找到 2048 点云中可表示最大特征值的一点
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # [bs, 128/2048]，获得的是edge feature的全局平均值
        # x = torch.cat((x1, x2), 1)  # [bs, 256/4096]

        # x = x.view(-1, self.output_size)
        
        # return x # [bs, 4096]
        #----------****实验6****----------

        #----------****实验7****----------
        coor, x = fps_downsample(coor, x, 512)
        coor, x = fps_downsample(coor, x, 256)
        coor, x = fps_downsample(coor, x, 128) # coor: [bs, 3, 128], f: [bs, 4096, 128]
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [bs, 128/2048]，npoints 维度归1，特征被聚合，找到 2048 点云中可表示最大特征值的一点
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # [bs, 128/2048]，获得的是edge feature的全局平均值
        x = torch.cat((x1, x2), 1)  # [bs, 256/4096]

        f = x.view(-1, self.output_size) # [bs, 4096]
        
        return coor, f # coor: [bs, 3, 128], f: [bs, 4096]
        #----------****实验7****----------



class PointNetfeat(nn.Module):
    """
    input
    - point_cloud: b x num_dims x npoints_1

    output
    - feture:  b x feature_size
    """

    def __init__(
        self, num_points=16382, global_feat=True, use_SElayer=False, hide_size=4096
    ):
        super(PointNetfeat, self).__init__()
        self.use_SElayer = use_SElayer
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, hide_size, 1)
        self.hide_size = hide_size
        if use_SElayer:
            self.se1 = SELayer1D(channel=64)
            self.se2 = SELayer1D(channel=128)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(hide_size)

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]  # x: [batch_size, 3, num_points]
        if self.use_SElayer:
            x = F.relu(self.se1(self.bn1(self.conv1(x))))
            x = F.relu(self.se2(self.bn2(self.conv2(x))))
            x = self.bn3(self.conv3(x))
        else:
            x = F.relu(self.bn1(self.conv1(x)))  # x: [batch_size, 64, num_points]
            x = F.relu(self.bn2(self.conv2(x)))  # x: [batch_size, 128, num_points]
            x = self.bn3(self.conv3(x))  # x: [batch_size, 1024, num_points]
        x, _ = torch.max(x, 2)  # x: [batch_size, num_points]
        x = x.view(-1, self.hide_size)
        return x


class PointNetRes(nn.Module):
    """
    input:
    - inp: b x (num_dims+id) x num_points

    outputs:
    - out: b x num_dims x num_points
    """

    def __init__(self, use_SElayer: bool = False):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)

        self.use_SElayer = use_SElayer
        if use_SElayer:
            self.se1 = SELayer1D(channel=64)
            self.se2 = SELayer1D(channel=128)
            self.se4 = SELayer1D(channel=512)
            self.se5 = SELayer1D(channel=256)
            self.se6 = SELayer1D(channel=128)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        npoints = x.size()[2]
        # x: [batch_size, 4, num_points]
        if self.use_SElayer:
            x = F.relu(self.se1(self.bn1(self.conv1(x))))
        else:
            x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x  # [batch_size, 64, num_points]

        if self.use_SElayer:
            x = F.relu(self.se2(self.bn2(self.conv2(x))))
        else:
            x = F.relu(self.bn2(self.conv2(x)))

        x = self.bn3(self.conv3(x))  # [batch_size, 1024, num_points]
        x, _ = torch.max(x, 2)  # [batch_size, 1024]
        x = x.view(-1, 1024)  # [batch_size, 1024]
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)  # [batch_size, 1024, num_points]
        x = torch.cat([x, pointfeat], 1)  # [batch_size, 1088, num_points]
        if self.use_SElayer:
            x = F.relu(self.se4(self.bn4(self.conv4(x))))
            x = F.relu(self.se5(self.bn5(self.conv5(x))))
            x = F.relu(self.se6(self.bn6(self.conv6(x))))
        else:
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x))  # [batch_size, 3, num_points]
        return x


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
        self.input_trans = nn.Conv1d(3, 4, 1)
        self.conv1 = torch.nn.Conv2d(8, 64, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=1, bias=False)
        #self.conv3 = torch.nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.conv3 = torch.nn.Conv2d(256, 768, kernel_size=1, bias=False)
        #self.conv4 = torch.nn.Conv2d(2176, 512, kernel_size=1, bias=False)
        self.conv4 = torch.nn.Conv2d(1664, 384, kernel_size=1, bias=False)

        #self.conv5 = torch.nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv5 = torch.nn.Conv2d(768, 256, kernel_size=1, bias=False)

        self.conv6 = torch.nn.Conv2d(512, 128, kernel_size=1, bias=False)

        # debug,增加conv8获得[bs, 128, 128]的中心点特征 f        
        self.conv7 = torch.nn.Conv2d(256, 3, kernel_size=1, bias=False)
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
        #self.bn3 = torch.nn.BatchNorm2d(1024)
        self.bn3 = torch.nn.BatchNorm2d(768)
        #self.bn4 = torch.nn.BatchNorm2d(512)
        self.bn4 = torch.nn.BatchNorm2d(384)
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.bn7 = torch.nn.BatchNorm2d(3)
        self.bn8 = torch.nn.BatchNorm2d(128)

        self.th = nn.Tanh()

    def forward(self, x):
        coor = x # [bs, 3, 2048]
        x = self.input_trans(x) # bs 3 np -> bs 4 np(2048)

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
        x = self.bn3(self.conv3(x))  # [batch_size, 768/1024, num_points, k]
        x = x.max(dim=-1, keepdim=False)[0]  # [bs, 768/1024, num_points]

        x, _ = torch.max(x, 2)  # [batch_size, 768/1024]
        #x = x.view(-1, 1024)  # [batch_size, 1024]
        x = x.view(-1, 768)  # [batch_size, 768]
        #x = x.view(-1, 1024, 1).repeat(1, 1, npoints)  # [batch_size, 768/1024, num_points]
        x = x.view(-1, 768, 1).repeat(1, 1, npoints)  # [batch_size, 768/1024, num_points]
        x = torch.cat([x, pointfeat], 1)  # [batch_size, 832/1088, num_points]

        if self.use_SElayer:
            x = get_graph_feature(x, k=self.k)  # [bs, 1664/2176, num_points, k]
            x = F.relu(self.se4(self.bn4(self.conv4(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 384/512, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 768/1024, num_points, k]
            x = F.relu(self.se5(self.bn5(self.conv5(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 256, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 512, num_points, k]
            x = F.relu(self.se6(self.bn6(self.conv6(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        else:
            x = get_graph_feature(x, k=self.k)  # [bs, 1664/2176, num_points, k]
            x = F.relu(self.bn4(self.conv4(x))) # [bs, 384/512, num_point, k]
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 384/512, num_points]

            # 1. DGCNN 的方式
            # --------*******-------
            coor_q, x_q = fps_downsample(coor, x, 512)
            x = get_graph_feature(x_q, k=self.k)  # [bs, 768/1024, num_points, k]
            # --------*******-------

            # 2. 未调整输入点云数：2048
            # --------*******-------
            # x = get_graph_feature(x, k=self.k)  # [bs, 768/1024, num_points, k]

            x = F.relu(self.bn5(self.conv5(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 256, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 512, num_points, k]
            x = F.relu(self.bn6(self.conv6(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        
        # 1. DGCNN 的方式应用fps下采样获得[bs, 256, num_points]
        # --------*******-------
        coor_q, x_q = fps_downsample(coor_q, x, 128)
        x = get_graph_feature(x_q, k=self.k)  # [bs, 256, num_points, k]
        # --------*******-------

        # 2. 未调整输入点云数：2048
        # --------*******-------
        # x = get_graph_feature(x, k=self.k)  # [bs, 512, num_points, k]

        # 调整卷积层数和参数的设置的方式
        # --------*******-------
        coor = self.th(self.bn7(self.conv7(x))) # [bs, 3, num_points, k] 
        coor = coor.max(dim=-1, keepdim=False)[0]  # [bs, 3, num_points]

        f = F.relu(self.bn8(self.conv8(x))) # [bs, 128, num_points, k]，这里将激活函数nn.Tanh()修改为：F.relu
        f = f.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        # --------*******-------
        
        # -----如果实验效果可以，尝试将BatchNorm2d改为GroupNorm，将relu改为LeakyReLU------
        return coor, f # [bs, 3, num_points]，[bs, 128, num_points]


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
        return x * y.expand_as(x) # 这里是计算 η 可校准获得edge feature y（促进较好的局部特征的生成），SELayer计算所得的F_2需要和原输入x相乘，可结合CAE图示理解

class SELayer1D(nn.Module):
    """
    input:
        x:(b, c, m)

    output:
        out:(b, c, m')
    """

    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        (b, c, _) = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


if __name__=='__main__':
    input = torch.randn(2,3,2048).to('cuda') # bs = 64, N = 2048
    # input = torch.randn(2,3,2048) # bs = 64, N = 2048 # CPU 运算

    #----------****实验5****----------
    # hide_size = 2048 // 4
    # output_size = 1024 // 4
    # use_SElayer = True
    # # "Pointfeat"，这里是选择使用的特征提取模块，Residualnet是基于EdgeConvResFeat，Pointfeat 基于 Pointfeat
    # encode = "Residualnet"
    # bottleneck_size = 4096
    # k = 16
    # pre_encoder = SpareNetEncode(
    #                             hide_size=hide_size,
    #                             output_size=output_size,
    #                             bottleneck_size=bottleneck_size,
    #                             use_SElayer=use_SElayer,
    #                             encode=encode,
    #                             k = 16,
    #                             ).to('cuda')
    # coor, f = pre_encoder(input)
    #----------****实验5****----------

    #----------****实验6****----------
    hide_size = 2048
    output_size = 4096
    use_SElayer = True
    # "Pointfeat"，这里是选择使用的特征提取模块，Residualnet是基于EdgeConvResFeat，Pointfeat 基于 Pointfeat
    encode = "Residualnet"
    bottleneck_size = 4096
    k = 16
    pre_encoder = SpareNetEncode(
                                hide_size=hide_size,
                                output_size=output_size,
                                bottleneck_size=bottleneck_size,
                                use_SElayer=use_SElayer,
                                encode=encode,
                                k = k,
                                ).to('cuda')
    # pre_encoder = SpareNetEncode(
    #                             hide_size=hide_size,
    #                             output_size=output_size,
    #                             bottleneck_size=bottleneck_size,
    #                             use_SElayer=use_SElayer,
    #                             encode=encode,
    #                             )                         
    coor, f = pre_encoder(input)
    #----------****实验6****----------

    print(coor.size(), f.size())