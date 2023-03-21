import torch
#import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN

knn = KNN(k=16, transpose_mode=False)


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


# CMLP
class Convlayer(nn.Module):
    def __init__(self,point_scales):
        super(Convlayer,self).__init__()
        self.point_scales = point_scales
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3)) # 这里卷积核尺寸为：1x3，有别于1x1
        self.conv2 = torch.nn.Conv2d(64, 64, 1)
        self.conv3 = torch.nn.Conv2d(64, 128, 1)
        self.conv4 = torch.nn.Conv2d(128, 256, 1)
        self.conv5 = torch.nn.Conv2d(256, 512, 1)
        self.conv6 = torch.nn.Conv2d(512, 1024, 1)
        self.maxpool = torch.nn.MaxPool2d((self.point_scales, 1), 1) # kernel_size:[2048,1]，表示将[2048,1]的tensor 找到一个最大值，故二维变一维
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)
    def forward(self,x):
        x = torch.unsqueeze(x,1) # 此处N=2048，可为512、256 x:[64, 2048, 3]->[64, 1, 2048, 3] 由于这里是二维卷积，因此所对应的维数：[B,C,W,H]，卷积操作的过程中会将卷积核转置后再作矩阵乘法
        x = F.relu(self.bn1(self.conv1(x))) # [64, 64, 2048, 1]
        x = F.relu(self.bn2(self.conv2(x))) # [64, 64, 2048, 1]

        x_128 = F.relu(self.bn3(self.conv3(x))) # [64, 128, 2048, 1]
        x_256 = F.relu(self.bn4(self.conv4(x_128))) # [64, 256, 2048, 1]
        x_512 = F.relu(self.bn5(self.conv5(x_256))) # [64, 512, 2048, 1]
        x_1024 = F.relu(self.bn6(self.conv6(x_512))) # [64, 1024, 2048, 1]

        x_128 = torch.squeeze(self.maxpool(x_128),2) # [64, 128, 2048, 1]->[64, 128, 1, 1]->[64, 128, 1]
        x_256 = torch.squeeze(self.maxpool(x_256),2) # [64, 256, 2048, 1]->[64, 256, 1, 1]->[64, 256, 1]
        x_512 = torch.squeeze(self.maxpool(x_512),2) # [64, 512, 2048, 1]->[64, 512, 1, 1]->[64, 512, 1]
        x_1024 = torch.squeeze(self.maxpool(x_1024),2) # [64, 1024, 2048, 1]->[64, 1024, 1, 1]->[64, 1024, 1]
        
        L = [x_1024,x_512,x_256,x_128]
        x = torch.cat(L,1) # [64, 1920, 1],将上述卷积层(conv3、conv4、conv5、conv6)中128、256、512、1024的输出特征级联
        return x  # [64, 1920, 1]

class Latentfeature(nn.Module):
    def __init__(self,num_scales,each_scales_size,point_scales_list):
        super(Latentfeature,self).__init__()
        self.num_scales = num_scales
        self.each_scales_size = each_scales_size
        self.point_scales_list = point_scales_list
        self.Convlayers1 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[0]) for i in range(self.each_scales_size)])
        self.Convlayers2 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[1]) for i in range(self.each_scales_size)])
        self.Convlayers3 = nn.ModuleList([Convlayer(point_scales = self.point_scales_list[2]) for i in range(self.each_scales_size)])
        self.conv1 = torch.nn.Conv1d(3,1,1)       
        self.bn1 = nn.BatchNorm1d(1)



    def forward(self,x):
        coor = x
        f = x
        x_1, _ = fps_downsample(coor, f, 512)
        x_2, _ = fps_downsample(coor, f, 256)
        x = [x, x_1, x_2]


        outs = []
        for i in range(self.each_scales_size):
            outs.append(self.Convlayers1[i](x[0])) # 将[64, 2048, 3]执行卷积获得[64, 1920, 1]聚合特征
        for j in range(self.each_scales_size):
            outs.append(self.Convlayers2[j](x[1])) # 将[64, 512, 3]执行卷积获得[64, 1920, 1]聚合特征
        for k in range(self.each_scales_size):
            outs.append(self.Convlayers3[k](x[2])) # 将[64, 256, 3]执行卷积获得[64, 1920, 1]聚合特征
        
        latentfeature = torch.cat(outs,2) # [64, 1920, 3], 将2048、512、128 分别获得的Latent Vector级联concate获得最终的特征
        #------******实验7******--------
        # 将这里的 latentfeature 视为 [B, N, C] 直接对其进行 fps 处理获得[bs, 3, num_points]，[bs, 128, num_points]
        # 这个实验推测成功率不高，因为latentfeature 表示纯高维特征，需要进行特征生成来获得点云
        # coor, f = fps_downsample
        #------******实验7******--------
        latentfeature = latentfeature.transpose(1,2) # [64, 3, 1920]
        latentfeature = F.relu(self.bn1(self.conv1(latentfeature))) # [64, 1, 1920]
        latentfeature = torch.squeeze(latentfeature,1) # [64, 1920]
        return latentfeature # [64, 1920]

class _netG(nn.Module):
    def  __init__(self,num_scales,each_scales_size,point_scales_list,crop_point_num):
        super(_netG,self).__init__()
        self.crop_point_num = crop_point_num
        self.latentfeature = Latentfeature(num_scales,each_scales_size,point_scales_list)
        self.fc1 = nn.Linear(1920,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        
        self.fc1_1 = nn.Linear(1024,128*512)
        self.fc2_1 = nn.Linear(512,64*128)#nn.Linear(512,64*256) !
        self.fc3_1 = nn.Linear(256,64*3)
        
#        self.bn1 = nn.BatchNorm1d(1024)
#        self.bn2 = nn.BatchNorm1d(512)
#        self.bn3 = nn.BatchNorm1d(256)#nn.BatchNorm1d(64*256) !
#        self.bn4 = nn.BatchNorm1d(128*512)#nn.BatchNorm1d(256)
#        self.bn5 = nn.BatchNorm1d(64*128)
#        
        self.conv1_1 = torch.nn.Conv1d(512,512,1)#torch.nn.Conv1d(256,256,1) !
        self.conv1_2 = torch.nn.Conv1d(512,256,1)
        # 更改“crop_point_num”以控制生成缺失点的数量，defalut = 512
        self.conv1_3 = torch.nn.Conv1d(256,int((self.crop_point_num*3)/128),1)
        self.conv2_1 = torch.nn.Conv1d(128,6,1)#torch.nn.Conv1d(256,12,1) !
        
#        self.bn1_ = nn.BatchNorm1d(512)
#        self.bn2_ = nn.BatchNorm1d(256)
        
    def forward(self,x):
        #------*****实验 8 *****-------
        # 仿照上述代码，去获得coor:[bs, 3, num_points]，f:[bs, 128, num_points], 去加入PoinTr执行实验
        # 本质上就是对 feature 执行 MLP->reshape 来获得目标点云
        x = self.latentfeature(x) # Final Feature Vector V： [64, 1920],因此输入 x 需要应用fps来获得2048、512、128 的下采样点云作为输入
        x_1 = F.relu(self.fc1(x)) # FC1，[64(bs), 1024]
        x_2 = F.relu(self.fc2(x_1)) # FC2，[64, 512]
        x_3 = F.relu(self.fc3(x_2))  # FC3，MLP，[64, 256]

        pc1_feat = self.fc3_1(x_3) # [64, 192]，可以改成[64, 192]
        pc1_xyz = pc1_feat.reshape(-1,64,3) #(M_1, 3) [64, 64, 3]由 FC3 生成的 Y_primary, 整体的中心center1
        
        


        return pc1_xyz
        #------*****实验 8 *****-------




if __name__=='__main__':
    input1 = torch.randn(64,2048,3) # bs = 64, N = 2048
    input2 = torch.randn(64,512,3)
    input3 = torch.randn(64,256,3)
    input_ = [input1,input2,input3]
    netG = _netG(3,1,[2048,512,256])
    output = netG(input_)
    print(output)