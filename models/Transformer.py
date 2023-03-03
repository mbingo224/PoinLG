import torch
import torch.nn as nn

from timm.models.layers import DropPath,trunc_normal_

from .dgcnn_group import DGCNN_Grouper
from utils.logger import *
import numpy as np
from knn_cuda import KNN
knn = KNN(k=8, transpose_mode=False)

# 这里是为每一个选取的中心点(N = 128)选取其周围的K个近邻点
def get_knn_index(coor_q, coor_k=None):
    coor_k = coor_k if coor_k is not None else coor_q
    # coor: bs, 3, np
    batch_size, _, num_points = coor_q.size()
    num_points_k = coor_k.size(2) # 将np赋给num_points_k

    with torch.no_grad():
        # 这里下划线变量返回的是每个中心点的近邻点的距离，由于 k = 8，因此其shape：bs x 8 x np(128)
        _, idx = knn(coor_k, coor_q)  # bs k np 这里是由于KNN继承的nn.Module定义了__call__方法了，由于forward 是 __call__的重名，因此调用knn(coor_k, coor_q)实际上是调用__call__方法，即等同于调用 forward 方法
        idx_base = torch.arange(0, batch_size, device=coor_q.device).view(-1, 1, 1) * num_points_k # view(-1, 1, 1) 表示将tensor重构为dim0 x 1 x 1，第0维由于是-1，将会自动补齐
        idx = idx + idx_base
        idx = idx.view(-1) # 这里是将idx从三维降为1维，相当于展平flaten，变成N x K = 8 x 128 = 1024 的索引范围
    
    return idx  # bs*k*np

# knn 几何感知模块获得包含局部特征和全局特征的输出feature
def get_graph_feature(x, knn_index, x_q=None):

        # x: bs, np, c, test时即1 128 384， knn_index: bs*k*np，即 1 8 128
        k = 8
        batch_size, num_points, num_dims = x.size()
        num_query = x_q.size(1) if x_q is not None else num_points
        # problem 索引时为什么不会超出范围，[1, 128, 384]->[1 * 128, 384]->[1024, 384]
        feature = x.view(batch_size * num_points, num_dims)[knn_index, :]
        # 这里为了合并相对位置feature（局部特征，实际上应该是 feature - x）和 绝对位置x（全局特征）
        # 将feature [1024, 384]的维度上升为[1, 8, 128, 384] 和 将x [1, 128, 384] 提升为[1, 8, 128, 384]
        feature = feature.view(batch_size, k, num_query, num_dims) # [bs, 8, 128, 384]
        x = x_q if x_q is not None else x
        x = x.view(batch_size, 1, num_query, num_dims).expand(-1, k, -1, -1) # 将输入 x 进行扩维，[1, 1 , 128, 384] -> [1, 8, 128, 384]
        
        # problem: 这里为什么局部特征还需要减去全局特征，再去做级联？ 原因是：
        # 这里是 KNN 
        feature = torch.cat((feature - x, x), dim=-1) # 扩维后的输入（全局特征）与 由knn计算得来的局部特征级联
        return feature  # b k np c，合并相对位置feature 和 绝对位置x,

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads # 采用的是多头注意力机制，这里是
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights 
        # 这里需要结合 Attention 的计算公式来理解，这里scale缩放因子过程就是除以head_dim ** -0.5，参考：
        self.scale = qk_scale or head_dim ** -0.5 # None or 0.125 = 0.125

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # 这里升维之所以是乘 3 是为获得三个输入 q k v
        self.attn_drop = nn.Dropout(attn_drop) # dropout 避免过拟合 e.g attn_drop = 0.0
        self.proj = nn.Linear(dim, dim) # 这里是对所求的注意力输出进行仿射变换
        self.proj_drop = nn.Dropout(proj_drop) # e.g proj_drop = 0.0

    def forward(self, x):
        B, N, C = x.shape
        # self.qkv(x): 1 x 128 x 384 -> 1 x 128 x 1152，因为这里需要获得 q、k、v三个输入，因此升维一个3，
        # 这里添加self.num_heads的目的是单头变多头，以便计算多头注意力，后面两个维度是由 384 根据多头的数量 num_heads = 6 拆分出两个维度
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # qkv.shape: [3, 1, 6, 128, 64]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        # q.shape: [1, 6, 128, 64] k.shape: [1, 6, 128, 64] @：矩阵乘法，需要保证倒数两维满足矩阵乘法的要求
        # q、k、v矩阵的组成形式为：[B, num_heads, N, dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale # 这是 Attention(Q, K, V)计算公式，乘以缩放因子（0.125）是为了避免数值太大导致 softmax 梯度消失
        attn = attn.softmax(dim=-1) # 对倒数第一维进行softmax运算，即对列的维度进行运算，使得每一列的元素的所有和为0
        attn = self.attn_drop(attn) # dropout problem：为什么概率设置为0

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # 获得级联了各个头（层面信息）的输出
        x = self.proj(x)
        x = self.proj_drop(x)
        return x # 返回级联了各个头（层面信息）的输出，shape:[1, 128, 384]



class CrossAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, v):
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, dim_q = None, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.self_attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        dim_q = dim_q or dim
        self.norm_q = norm_layer(dim_q)
        self.norm_v = norm_layer(dim)
        self.attn = CrossAttention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim*2, dim)

        self.knn_map_cross = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map_cross = nn.Linear(dim*2, dim)

    def forward(self, q, v, self_knn_index=None, cross_knn_index=None):
        # q = q + self.drop_path(self.self_attn(self.norm1(q)))
        norm_q = self.norm1(q)
        q_1 = self.self_attn(norm_q)

        if self_knn_index is not None:
            knn_f = get_graph_feature(norm_q, self_knn_index)
            knn_f = self.knn_map(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_1 = torch.cat([q_1, knn_f], dim=-1)
            q_1 = self.merge_map(q_1)
        
        q = q + self.drop_path(q_1)

        norm_q = self.norm_q(q)
        norm_v = self.norm_v(v)
        q_2 = self.attn(norm_q, norm_v)

        if cross_knn_index is not None:
            knn_f = get_graph_feature(norm_v, cross_knn_index, norm_q)
            knn_f = self.knn_map_cross(knn_f)
            knn_f = knn_f.max(dim=1, keepdim=False)[0]
            q_2 = torch.cat([q_2, knn_f], dim=-1)
            q_2 = self.merge_map_cross(q_2)

        q = q + self.drop_path(q_2)

        # q = q + self.drop_path(self.attn(self.norm_q(q), self.norm_v(v)))
        q = q + self.drop_path(self.mlp(self.norm2(q)))
        return q


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim) # nn.LayerNorm是在 N(batch_size)方向做归一化，即针对C、H、W通道计算均值和方差，与batch无关
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 这里nn.Identity()输入是什么输出就是什么，即只是增加一层无用层而以
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.knn_map = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LeakyReLU(negative_slope=0.2)
        )

        self.merge_map = nn.Linear(dim*2, dim)

        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, knn_index = None):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        norm_x = self.norm1(x) # 计算注意力前先进行层归一化，x.shape: 1 x 128(N) x 384(C) layer归一化，这里由于点云数据的特殊性，因此只对 N x C 作归一化，这里的N就相当于图像数据的H X W
        x_1 = self.attn(norm_x) # 计算多头自注意力，norm_x.shape:[1, 128, 384]

        if knn_index is not None: # 第一层 knn计算，即引入的另一个输入获得几何感知，相当于encorder的第一层有两个输入，一个是常规的q、k、v计算注意力
            knn_f = get_graph_feature(norm_x, knn_index) # 合并相对位置feature 和 绝对位置x
            knn_f = self.knn_map(knn_f) # 线性映射后降维为 [1, 128, 384]
            # 按维度dim 返回最大值，这里是返回每一行的最大值，最大值和索引各是一个tensor，
            # torch.max()[0]: 只返回最大值，不返回索引，keepdim表示是否按照原输入的维度形式输出，默认False
            knn_f = knn_f.max(dim=1, keepdim=False)[0] # knn几何感知模块获得的特征在

            # 将KNN所获得的特征knn_f 与经过 encorder 第一层多头注意力计算的x_1 在最后一个维度上进行级联
            # Encoder 第一层的两个输入的级联
            x_1 = torch.cat([x_1, knn_f], dim=-1) # x_1.shape: [bs, 128, 768]
            x_1 = self.merge_map(x_1) # 映射到原始维度，x_1.shape:[1, 128, 384]
        
        x = x + self.drop_path(x_1) # dropout 放置在后面效果更好
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



class PCTransformer(nn.Module):
    """ Vision Transformer with support for point cloud completion
    """
    def __init__(self, in_chans=3, embed_dim=768, depth=[6, 6], num_heads=6, mlp_ratio=2., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                        num_query = 224, knn_layer = -1):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim # 特征数的初始化
        
        self.knn_layer = knn_layer # KNN层数的设置

        print_log(' Transformer with knn_layer %d' % self.knn_layer, logger='MODEL')

        # 使用DGCNN进行特征提取，DGCNN结合了全局和局部特征，考虑每个点本身（全局特征）以及周围点与其的相对距离（近邻点）
        # 可参考：https://zhuanlan.zhihu.com/p/425724743 中解释
        self.grouper = DGCNN_Grouper()  # B 3 N to B C(3) N(128) and B C(128) N(128)

        # 这里是获取position_embeding来恢复点云输入序列中的时序信息，这里采用的是一维卷积，因为Transformer就是专门用来处理文本这种一维数据
        # 批归一化的参数选择一般上一输入层形状的N x C 中的 C，输出形状和输入形状是一致的
        # nn.LeakyReLU()函数是ReLU的弱化版，negative_slope控制负斜率的大小，负数时斜率不再是0，默认值：1e-2
        # 提升通道数，可以丰富特征，容易将点与点之间区分，但是也不能太大，会淡化点与点之间的关系，生成各个维度的位置信息，当输入中两个位置发生交换时，使得模型能感知这种变化，弥补自注意力机制不能捕捉序列时序信息
        # 通道数 C：3->128->384
        self.pos_embed = nn.Sequential(
            nn.Conv1d(in_chans, 128, 1), # NOTE: Conv1d 只对 B C N 三个维度中的 C 维度进行计算
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2), # 这样负数时不会全部归置为0
            nn.Conv1d(128, embed_dim, 1) # 这里就是相当于MLP(全连接层)
        )
        # self.pos_embed_wave = nn.Sequential(
        #     nn.Conv1d(60, 128, 1),
        #     nn.BatchNorm1d(128),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(128, embed_dim, 1)
        # )

        # 这部分最后一层使用1 x 1 卷积可以增加非线性同时也保证了输出维度不变
        # 使用 nn.Sequential模块将会自动实现forward方法
        # 使用 nn.Parameter可以让这个cls_token作为一个像权重那样的可学习参数，随着网络的训练不断的调优
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 通道数 C：128->384->384
        self.input_proj = nn.Sequential(
            nn.Conv1d(128, embed_dim, 1),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(embed_dim, embed_dim, 1)
        )
        
        # 构建Transformer的几何感知的Encoder模块，这里使用nn.ModuleList，使得可以在forward函数中自定义一些操作
        self.encoder = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[0])])

        # self.increase_dim = nn.Sequential(
        #     nn.Linear(embed_dim,1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024)
        # )
        # 通过对比发现利用这种巻积这种方式去升维的方式相比使用线性层的方式，网络相对更深，且非线性化程度更好
        # 查阅可知：对前层是全连接的全连接层可以转化为卷积核为1x1的卷积，这样可以减少参数，但模型的迁移性减弱
        self.increase_dim = nn.Sequential(
            nn.Conv1d(embed_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )

        self.num_query = num_query
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True), # 表示原地执行，表示新计算的结果将会覆盖原值，达到节约内存的效果
            nn.Linear(1024, 3 * num_query)
        )
        self.mlp_query = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1),
            # nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, embed_dim, 1)
        )

        self.decoder = nn.ModuleList([
            DecoderBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth[1])])

        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.cls_pos, std=.02)
        self.apply(self._init_weights) # 使用apply 来初始化模块和子模块的参数

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_normal_(m.weight.data, gain=1)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight.data, 1)
            nn.init.constant_(m.bias.data, 0)

    def pos_encoding_sin_wave(self, coor):
        # ref to https://arxiv.org/pdf/2003.08934v2.pdf
        D = 64 #
        # normal the coor into [-1, 1], batch wise
        normal_coor = 2 * ((coor - coor.min()) / (coor.max() - coor.min())) - 1 

        # define sin wave freq
        freqs = torch.arange(D, dtype=torch.float).cuda() 
        freqs = np.pi * (2**freqs)       

        freqs = freqs.view(*[1]*len(normal_coor.shape), -1) # 1 x 1 x 1 x D
        normal_coor = normal_coor.unsqueeze(-1) # B x 3 x N x 1
        k = normal_coor * freqs # B x 3 x N x D
        s = torch.sin(k) # B x 3 x N x D
        c = torch.cos(k) # B x 3 x N x D
        x = torch.cat([s,c], -1) # B x 3 x N x 2D
        pos = x.transpose(-1,-2).reshape(coor.shape[0], -1, coor.shape[-1]) # B 6D N
        # zero_pad = torch.zeros(x.size(0), 2, x.size(-1)).cuda()
        # pos = torch.cat([x, zero_pad], dim = 1)
        # pos = self.pos_embed_wave(x)
        return pos

    def forward(self, inpc):
        '''
            inpc : input incomplete point cloud with shape B N(2048) C(3)
        '''
        # build point proxy，inpc表示输入点云，形状为：B x N x C
        bs = inpc.size(0) # 查看第 0 维的大小，即 B：batch_size

        # self.grouper = DGCNN_Grouper()，coor的shape:B x C(3) x N(128)，f 的shape：B x C(128) x N(128)
        # 特征 f 的通道 C 会得到提升
        coor, f = self.grouper(inpc.transpose(1,2).contiguous()) # DGCNN 分别得到中心点坐标coor及中心点特征f，输入点云被转置为B x C x N，contiguous是保证inpc转置以后保证底层数据从不连续转变为连续的
        # 获得N个中心点中每个点的K个近邻点的索引，shape: [k*N]，e.g 也就是 8 * 128 个近邻点的距离
        knn_index = get_knn_index(coor)

        # NOTE: try to use a sin wave  coor B 3 N, change the pos_embed input dim
        # pos = self.pos_encoding_sin_wave(coor).transpose(1,2)
        # 对于中心点坐标coor进行位置编码，shape: B x C(3) x N(128) -> B x C(384) x N(128)-> B x 128 x 384，将其通道数提升
        # 位置编码的意义在于保证中心点经 encorder 后的特征仍然保有原有输入的序列信息
        # NOTE：pos embeding是对最后一个维度（128个点）操作的，即计算所有点与点之间的位置关系，
        pos =  self.pos_embed(coor).transpose(1,2) # 使中心点的坐标通过 MLP 对位置编码

        # 中心点特征f 进行维度变换，变换为与中心点坐标相同的维度，以求和来作为编码器的输入
        # f: 1 x 128 x 128 -> 1 x 128(C) x 384(N)
        # NOTE：这里有一个problem: 是否可以将 x + pos后再执行 pos_embeding 是否会影响收敛的效果，因为可以认为特征不需要
        x = self.input_proj(f).transpose(1,2) # 特征通过 MLP，以便于与pos 级联形成点代理
        
        # expand 仅在维度为1上执行bs次重复操作，以达到升高维数的效果
        # cls_pos = self.cls_pos.expand(bs, -1, -1)
        # cls_token = self.cls_pos.expand(bs, -1, -1)
        # x = torch.cat([cls_token, x], dim=1)
        # pos = torch.cat([cls_pos, pos], dim=1)
        # encoder
        for i, blk in enumerate(self.encoder):
            if i < self.knn_layer: # 由于self.knn_layer = 1，因此只在Encoder 和 Decorder的第一层进行attention的注意力计算和中心点相对位置编码的整合，然后继续输入后续网络
                # 中心点坐标与中心点特征求和作为计算多头注意力的输入，
                # 并且还对输入执行 knn 几何感知获得局部特征与全局特征
                x = blk(x + pos, knn_index)   # B N C，这里是图示中DGCNN后第一次求和
            else:
                x = blk(x + pos) # i = [1, ... , 5]，不进行 knn 计算

        # build the query feature for decoder
        # global_feature  = x[:, 0] # B C

        global_feature = self.increase_dim(x.transpose(1,2)) # B N 1024 -> B 1024 N（转置），线性投射升维
        # 理论上最大池化虽然可以获得全局特征，但是会不可避免的导致局部细节特征的丢失
        #  max 对称操作使得输入点的顺序不会对模型预测产生影响，保证置换不变性的有效
        global_feature = torch.max(global_feature, dim=-1)[0] # B 1024，

        coarse_point_cloud = self.coarse_pred(global_feature).reshape(bs, -1, 3)  #  B M C(3)

        new_knn_index = get_knn_index(coarse_point_cloud.transpose(1, 2).contiguous())
        cross_knn_index = get_knn_index(coor_k=coor, coor_q=coarse_point_cloud.transpose(1, 2).contiguous())

        query_feature = torch.cat([
            global_feature.unsqueeze(1).expand(-1, self.num_query, -1), 
            coarse_point_cloud], dim=-1) # B M C+3 
        q = self.mlp_query(query_feature.transpose(1,2)).transpose(1,2) # B M C 
        # decoder
        for i, blk in enumerate(self.decoder):
            if i < self.knn_layer:
                q = blk(q, x, new_knn_index, cross_knn_index)   # B M C
            else:
                q = blk(q, x)

        return q, coarse_point_cloud

