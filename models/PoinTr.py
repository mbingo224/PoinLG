import torch
from torch import nn

from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1
from .Transformer import PCTransformer
from .build import MODELS


def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc


class Fold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel # =trans_dim = in_channel = 384
        self.step = step

        # a: 构建[-1, 1]的长度为 step 的一维张量[step]->[1, step]->[8, 8]->[1, 64]，创建 2D grid，原 foldingnet 是由numpy来创建的
        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        # 构建 2D grids
        self.folding_seed = torch.cat([a, b], dim=0).cuda() # [2, 64]

        # 这里是 1st folding 与 2nd folding 模块的设计，对比原网络的FoldingLayer来分析，和原网络的self.layers是等效的
        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        # x.shape: [B*M, 384]，其中M=224
        num_sample = self.step * self.step # 这里 step 实际上是 2D grid的长和宽，原 FoldingNet 的 step = 45
        bs = x.size(0) # B*M
        # in_channel = 384，[B*M, 384, 1]->[bs, 384, 64]
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        # [2, 64]->[1, 2, 64]->[bs, 2, 64]，这里的self.folding_seed 即是 foldingnet 的 self.grid
        # 提升维度目的在于 batch size 一致，作为 seed 引导三维点云的还原生成
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device) # 利用GPU加速计算

        # 与FoldingNet 作为对比, 2D grid（[m, 2]，这里的 m 表示而为网格中 m 个像素点) 就是此处的self.folding_seed
        # 在此处，m 被假定为 64(m = step x step)，由 512(C)调整为 384，这是因为Transformer 的 decorder 所生成而来的，可查阅下述PointTr 的 forward 函数
        # 将seed 和 features 级联以后活动的特征x([bs, 384+2, 64])
        # 这里引入 2D grid 就是 foldingnet 将 384+2 高维度的点云特征投射回三维
        # NOTE: problem: step 的来源 意义值的思考一下？可以结合 FoldingNet 的 encorder 模块思考一下
        # answer:step来源于构建 2D grid的时候的grid 的长和宽
        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x) # fd1.shape:[bs, 384+3, 64]
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x) # [bs, 3, 64]

        return fd2

@MODELS.register_module()
class PoinTr(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.trans_dim = config.trans_dim # 为 embed_dim 初始化 384
        self.knn_layer = config.knn_layer # 1 knn 的层数
        self.num_pred = config.num_pred # 需要补全的点的数量，因为本模型只负责恢复缺失的点云(N = 14336)，最终再和原输入的点云(N = 2048)执行加和运算得到完整的点云(N = 14336 + 2048 = 16384)
        self.num_query = config.num_query # 224 这里num_query 表示对原始输入点云（残缺点云）进行采样的数值

        # 折叠步骤fold_step=8 由num_pred和num_query计算而来
        self.fold_step = int(pow(self.num_pred//self.num_query, 0.5) + 0.5) # 14336 // 224 = 64, 加 0.5 实现四舍五入，可以用round函数来代替，注意观察这里的depth，理论上Transformer的encorder和decorder均是6层，因此这里解码层改变为8层是否会对解码有促进作用，待验证
        self.base_model = PCTransformer(in_chans = 3, embed_dim = self.trans_dim, depth = [6, 8], drop_rate = 0., num_query = self.num_query, knn_layer = self.knn_layer)
        
        self.foldingnet = Fold(self.trans_dim, step = self.fold_step, hidden_dim = 256)  # rebuild a cluster point

        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.reduce_map = nn.Linear(self.trans_dim + 1027, self.trans_dim)
        self.build_loss_func()

    def build_loss_func(self):
        self.loss_func = ChamferDistanceL1()

    def get_loss(self, ret, gt):
        loss_coarse = self.loss_func(ret[0], gt)
        loss_fine = self.loss_func(ret[1], gt)
        return loss_coarse, loss_fine

    def forward(self, xyz):
        # 经过PCTransformer模块计算得到粗糙点云coarse_point_cloud（bs, 224, 3）和 decorder 输出的包含全局和局部特征的注意力predicted proxies: q（bs, 224, 384）
        q, coarse_point_cloud = self.base_model(xyz) # B M C and B M 3，对输入的partial部分点去进行FPS、MLP、DGCNN，然后position_embeding输入Transformer
    
        B, M ,C = q.shape
        # q.shape: B N(224) C(384)->B C(384) N(224)->B C(1024) N(224)->B N(224) C(1024) 对词向量的维度C作变换
        global_feature = self.increase_dim(q.transpose(1,2)).transpose(1,2) # B M 1024
        # 聚合操作这里用的是channel wise 的 max pooling，因为这里获得的是所有通道C中最大值的一个点
        global_feature = torch.max(global_feature, dim=1)[0] # B 1024，最大池化保证置换不变性，在哪一个维度上执行最大池化，这个维度就会消失
        
        # global_feature.shape: [1, 1024]->[1, 1, 1024]->[1, 224, 1024]
        # 细化模块foldingnet的输入：decorder-encorder 整合的特征global_feature、初始预测的位置特征粗糙点云coarse_point_cloud、
        # decorder-encorder 整合的特征未最大池化 q 级联获得，加入q 可能使细化效果更好
        rebuild_feature = torch.cat([
            global_feature.unsqueeze(-2).expand(-1, M, -1),
            q,
            coarse_point_cloud], dim=-1)  # B M(224) 1027 + C(1411)

        rebuild_feature = self.reduce_map(rebuild_feature.reshape(B*M, -1)) # BM C(384), 降低维数的全连接映射
        # # NOTE: try to rebuild pc
        # coarse_point_cloud = self.refine_coarse(rebuild_feature).reshape(B, M, 3)

        # NOTE: foldingNet
        # 将上述合并特征输入 FoldingNet 预测相对位置
        relative_xyz = self.foldingnet(rebuild_feature).reshape(B, M, 3, -1)    # B M(224) 3 S(64)
        # rebuild_points：[1, 14336, 3]，变成绝对位置rebuild_points，又再一次整合了粗糙点云输入，补充特征
        rebuild_points = (relative_xyz + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)  # B N 3

        # NOTE: fc
        # relative_xyz = self.refine(rebuild_feature)  # BM 3S
        # rebuild_points = (relative_xyz.reshape(B,M,3,-1) + coarse_point_cloud.unsqueeze(-1)).transpose(2,3).reshape(B, -1, 3)

        # cat the input，获得原始输入的 num_query 个采样点的稀疏点云
        inp_sparse = fps(xyz, self.num_query)
        # coarse_point_cloud：[1, 448, 3]，和预测中心点在点云数量维度上级联（224+224），有利于计算后续的 SparseLoss，有监督预测的粗糙点云
        coarse_point_cloud = torch.cat([coarse_point_cloud, inp_sparse], dim=1).contiguous()
        # rebuild_points：[1, 16384, 3]，原始点云与细化后重构的全部点云级联作为完整点云输出
        rebuild_points = torch.cat([rebuild_points, xyz],dim=1).contiguous()

        ret = (coarse_point_cloud, rebuild_points) # ([1, 448, 3]，[1, 16384, 3])
        return ret

