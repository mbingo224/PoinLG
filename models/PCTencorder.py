from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils
import math
# 由于单独运行这个文件时的工作目录为 models 目录，故添加PointAttN 目录到工作路径，保证导入和models同级的目录utils中的模块model_utils的work
import sys,os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
path = sys.path.append(BASE_DIR+"/../")


# import utils.mm3d_pn2.ops.furthest_point_sample.furthest_point_sample as furthest_point_sample
# import utils.mm3d_pn2.ops.gather_points.gather_points as gather_points

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

class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # 原始的transformer
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1) # [bs, 64, 512]
        src2 = self.input_proj(src2) # [bs, 64, 2048]

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1) # [512, bs, 64]
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1) # [2048, bs, 64]

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)
        # multi-head self-attention, 计算 query 和 key 的相似度，然后乘以value，k、v 来源于同一个输入 X，q 来源于 Q
        # [512, bs, 64]
        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]


        src1 = src1 + self.dropout12(src12) # [512, bs, 64] 正则化
        src1 = self.norm12(src1)

        # feed forward
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)


        src1 = src1.permute(1, 2, 0)

        return src1 # [8, 64, 512]


class PCT_refine(nn.Module):
    def __init__(self, channel=128,ratio=1):
        super(PCT_refine, self).__init__()
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = cross_transformer(channel*2,512)
        self.sa2 = cross_transformer(512,512)
        self.sa3 = cross_transformer(512,channel*ratio)

        self.relu = nn.GELU()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

        self.channel = channel

        self.conv_delta = nn.Conv1d(channel * 2, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel*ratio, channel*ratio, kernel_size=1)

        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)


    def forward(self, x, coarse,feat_g):
        batch_size, _, N = coarse.size()
        '''由corse粗糙点云和形状特征feat_g获得两个点特征，进行逐点级联，可使得y0保留了coarse的全局特征的同时也获得feat_g的局部形状细节信息'''
        y = self.conv_x1(self.relu(self.conv_x(coarse)))  # B, C, N [8, 3, 512]->[8, 128, 512]
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))  # B, C, N，一维卷积MLP [8, 512, 1]->[8, 128, 1]
        y0 = torch.cat([y,feat_g.repeat(1,1,y.shape[-1])],dim=1) # 坐标空间y与特征空间feat_g([8, 128, 1]->[8, 128, 512])在 通道维度级联生成y0:[8, 256, 512]
        '''NOTE: 这里和 seed generator 作用一致: 逐步对点特征进行上采样'''
        y1 = self.sa1(y0, y0) # [8, 512, 512]
        y2 = self.sa2(y1, y1) # [8, 512, 512]
        y3 = self.sa3(y2, y2) # [8, 512, 512]
        y3 = self.conv_ps(y3).reshape(batch_size,-1,N*self.ratio) # [8, 128, 2048]

        y_up = y.repeat(1,1,self.ratio) # [8, 128, 512]->[8, 128, 2048],因为self.ratio = 4，repeat操作 512 * 4
        y_cat = torch.cat([y3,y_up],dim=1) # [8, 256, 2048]
        y4 = self.conv_delta(y_cat) # 一维mlp [8, 128, 2048]
        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1,1,self.ratio) # [8, 3, 2048] + [8, 3, 2048]

        return x, y3 # 细化后的点云x：[8, 3, 2048]) 

class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel,channel)
        self.sa1_1 = cross_transformer(channel*2,channel*2)
        self.sa2 = cross_transformer((channel)*2,channel*2)
        self.sa2_1 = cross_transformer((channel)*4,channel*4)
        self.sa3 = cross_transformer((channel)*4,channel*4)
        self.sa3_1 = cross_transformer((channel)*8,channel*8)

        self.relu = nn.GELU()


        self.sa0_d = cross_transformer(channel*8,channel*8)
        self.sa1_d = cross_transformer(channel*8,channel*8)
        self.sa2_d = cross_transformer(channel*8,channel*8)

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel*4, 64, kernel_size=1)
        #----------****实验15****----------#
        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel*2, 64, kernel_size=1)
        #----------****实验15****----------#
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)


    def forward(self, points):
        batch_size, _, N = points.size() # points表示

        ''''
        NOTE: 这两行代码起到position embedding的作用, 以便Transformer可以区分不同位置之间的关系, 
        更好地理解输入序列中的局部信息, Transformer模型中使用了多层自注意力机制来整合局部信息, 以生成全局特征表示
        '''
        # 这里实际上是对输入点云进行了两次卷积，得到了一个 64 维的特征
        x = self.relu(self.conv1(points))  # B, D, N [bs(2), 64, 2048]
        x0 = self.conv2(x) # B, C, N [bs, 64, 2048] GDP 模块的输入 X，即输入点云经过两层卷积后得到的特征

        '''NOTE: GDP 模块计算的是交叉注意力 有利于获得点云的局部几何结构细节'''
        # GDP /4，这里是应用 fps 获得不同分辨率的 Y，由 Y 生成 Q，这里采样率 d = 4 N = 512
        # x_g0: [bs, 64, 512] 由输入点云生成64维的特征x0的下采样点云，得到 Y
        # points: [bs, 3, 512] 根据 idx_0 从输入点云中得到下采样点云，得到下一个GDP模块的输入 X1
        points, x_g0 = fps_downsample(points, x0, N // 4)
        
        '''
        NOTE: 这里的sa1与SFA模块采用的sa1_1在计算注意力上有差异, 前者由于输入的是两个不同序列, 因此实质上计算的是交叉注意力cross-attention
        , 后者由于输入是两个相同的输入序列, 因此计算的是自注意力self-attention
        '''
        x1 = self.sa1(x_g0, x0).contiguous() # [bs, 64, 512] 计算多头交叉注意力，根据 Y 和 X 经 SFA 模块生成 x1
        x1 = torch.cat([x_g0, x1], dim=1) # [bs, 128, 512] 将 Y 和 x1 拼接输出 output
        
        ''' SFA 1, NOTE: SFA 模块通过计算自己与自己的多头注意力(可降低计算复杂度),使得每个点更关注自身位置关系, 提取出更多的全局特征, 即 multi-head self-attention'''
        x1 = self.sa1_1(x1,x1).contiguous() # [bs, 128, 512] 经 SFA 模块生成 x1
        
        # GDP /2，points/2 这里是应用 fps 获得不同分辨率的 Y，由 Y 生成 Q，这里采样率 d = 8 N // 8 = 256
        # x_g1: [bs, 128, 256], 由输入点云生成128维的特征x1的下采样点云，得到 Y,
        # points: [bs, 3, 256] 根据 idx_1 从输入点云中得到下采样点云，得到下一个GDP模块的输入 X2
        points, x_g1 = fps_downsample(points, x1, N // 8)

        x2 = self.sa2(x_g1, x1).contiguous()  # C*2, N [bs, 128, 256]
        x2 = torch.cat([x_g1, x2], dim=1) # [bs, 256, 256]
        
        # SFA 2
        x2 = self.sa2_1(x2, x2).contiguous() # [bs, 256, 256] 
        
        # GDP /2 这里是应用 fps 获得不同分辨率的 Y，由 Y 生成 Q，这里采样率 d = 16 N // 16 = 128
        # x_g2: [bs, 256, 128]
        points, x_g2 = fps_downsample(points, x2, N // 16)

        x3 = self.sa3(x_g2, x2).contiguous()  # C*4, N/4 [bs, 256, 128]
        x3 = torch.cat([x_g2, x3], dim=1) #  [bs, 512, 128]
        
        # SFA 3
        x3 = self.sa3_1(x3,x3).contiguous() # [bs, 512, 128] 计算多头交叉注意力
        
        # maxpooling
        '''这里就是Feature extractor模块获得的特征编码shape code'''
        # adaptive_max_pool1d是一种自适应的池化方式，可以根据输入的数据自动调整池化核的大小，对序列的长度不敏感，改变序列长度到1 也可以为其他长度
        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1) # shape code: [bs, 512, 1]
        # NOTE：“逐点卷积层”：是一种“自我注意力机制”，即让每个通道都关注自己的信息，从而更好地捕捉特征之间的关系
        x = self.relu(self.ps_adj(x_g)) # [bs, 512, 1] 这里输出和输入的channel相同，使用1x1卷积的一维卷积
        x = self.relu(self.ps(x)) # NOTE：[bs, 64, 128] 应用反卷积，恢复到128个点云，这里有别于其他模型中先使用线性MLP 降维，再直接应用reshape获得128个3D位置坐标点云
        # NOTE：改进Tips: 设计 ps_refuse 的输出通道数为128，可获得中心点的特征shape:[bs, 128, 128]
        x = self.relu(self.ps_refuse(x)) # [bs, 512, 128] Conv1d 
        
        ''''Seed generator: 生成稀疏但完整的粗糙点云！'''
        # 串联3个SFA，增强感知目标几何结构的特征能力
        x0_d = (self.sa0_d(x, x)) # [bs, 512, 128] 经 SFA 模块生成 x0_d
        x1_d = (self.sa1_d(x0_d, x0_d)) # [bs, 512, 128] 经 SFA 模块生成 x1_d
        ''' 
        这里的特征reshape是为了对点特征进行MLP分割产生粗糙点坐标, 即将特征向量转化为3D位置坐标
        ***IDEA: 由于GDP和SFA模块已经聚合了全局和局部特征, 因此可以在这里通过reshape可以直接变换
        得到Transformer中的decorder的 q [bs, 224, 384], 即作为 x_g 输出, 如果这样就需要调整x_g2的下采样率, 由128改为168
        '''
        #x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size,self.channel*4,N//8) # [bs, 512, 128]->[bs, 256, 256]

        #----------****实验15****----------#
        x2_d = (self.sa2_d(x1_d, x1_d)).reshape(batch_size, 128, 512) # [bs, 512, 128]->[bs, 256, 256]
        
        coarse = self.conv_out(self.relu(self.conv_out1(x2_d))) # [bs, 3, 512]
        
        #----------****实验15****----------#


        # 这里使用的是一维卷积形式的MLP来生成粗糙点云fine，不同于其他网络使用Liner的MLP，可比对Transformeer.py
        coarse = self.conv_out(self.relu(self.conv_out1(x2_d))) # [bs, 3, 256]

        return points, x3, coarse # points: [bs, 3, 128] 形状编码x3: [bs, 512, 128] 粗糙点云fine: [bs, 3, 256(num_query)]

class Model(nn.Module):
    # def __init__(self, args):
    def __init__(self):
        super(Model, self).__init__()

        # if args.dataset == 'pcn':
        #     step1 = 4
        #     step2 = 8
        # elif args.dataset == 'c3d':
        #     step1 = 1
        #     step2 = 4
        # else:
        #     ValueError('dataset is not exist')

        self.encoder = PCT_encoder()

        self.increase_dim = nn.Sequential(
            nn.Conv1d(512, 1024, 1), # 
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.conv = nn.Conv1d(512, 384, kernel_size=1)
        # self.refine = PCT_refine(ratio=step1)
        # self.refine1 = PCT_refine(ratio=step2)
        self.relu = nn.GELU()
        self.ps_adj = nn.Conv1d(384, 384, kernel_size=1)

    def forward(self, x, num_query=224, gt=None, is_training=True):
        # feat_g 即是shape code(特征向量x_g):[bs, 512, 128], coarse 即是 fine(粗糙点云)：[bs, 3, 256], coor 即是 points(中心点坐标)：[bs, 3, 128]
        coor, feat_g, coarse = self.encoder(x)
        batch_size = coor.size(0)
        feat_x = feat_g
        feat_x = self.ps_adj(self.relu(self.conv(feat_x))).transpose(1,2) # 一维mlp [bs, 128, 384]
        
        feat_g = self.increase_dim(feat_g) # [bs, 1024, 128], 一维卷积是对第1维(通道)进行卷积，第2维(点数)不变
        feat_g = F.adaptive_max_pool1d(feat_g, 1).view(batch_size, -1)
        
        #----------****实验10****----------
        '''
        NOTE: 这里的 new_x 与 PoinTr 中的 coarse_point_cloud 生成是同样的思路,不过是先求和再fps
        这里新生成的 new_x 即作为 seeds 
        '''
        # new_x = torch.cat([x,coarse],dim=2) # [bs, 3, 256 + 2048 = 2304]
        # new_x, _ = fps_downsample(new_x, new_x, 1024)
        # new_x, _ = fps_downsample(new_x, new_x, num_query)
        # new_x = new_x.transpose(1, 2)
        #----------****实验10****----------

        #----------****实验11****----------
        #new_x = coarse.transpose(1, 2)
        #----------****实验11****----------

        #----------****实验14****----------效果很差
        # new_x, _ = fps_downsample(x, x, 128)
        # new_x = torch.cat([new_x,coarse],dim=2)
        # new_x = new_x.transpose(1, 2)
        #----------****实验14****----------
        #----------****实验15****----------
        new_x = coarse.transpose(1, 2)
        #----------****实验15****----------

        
        # coor(中心点坐标): [bs, 3, 128], feat_x(匹配原encorder的输出x): [bs, 128, trans_dim], 
        # feat_g(针对提取的特征变换维度获取注意力计算所得局部和全局特征): [bs, 1024], new_x(粗糙点云): [bs, num_query, 3]
        return coor, feat_x, feat_g, new_x 

        '''
        fine, feat_fine = self.refine(None, new_x, feat_g)
        fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)

        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        fine1 = fine1.transpose(1, 2).contiguous()

        if is_training:
            loss3, _ = calc_cd(fine1, gt)
            gt_fine1 = gather_points(gt.transpose(1, 2).contiguous(), furthest_point_sample(gt, fine.shape[1])).transpose(1, 2).contiguous()

            loss2, _ = calc_cd(fine, gt_fine1)
            gt_coarse = gather_points(gt_fine1.transpose(1, 2).contiguous(), furthest_point_sample(gt_fine1, coarse.shape[1])).transpose(1, 2).contiguous()

            loss1, _ = calc_cd(coarse, gt_coarse)

            total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()

            return fine, loss2, total_train_loss
        else:
            cd_p, cd_t = calc_cd(fine1, gt)
            cd_p_coarse, cd_t_coarse = calc_cd(coarse, gt)

            return {'out1': coarse, 'out2': fine1, 'cd_t_coarse': cd_t_coarse, 'cd_p_coarse': cd_p_coarse, 'cd_p': cd_p, 'cd_t': cd_t}
        '''

if __name__=='__main__':
    input = torch.randn(2,3,2048).to('cuda') # bs = 64, N = 2048
    encoder = Model().to('cuda')
    coor, feat_x, feat_g, coarse = encoder(input)
    print(coor.size(), feat_x.size(), feat_g.size(), coarse.size())