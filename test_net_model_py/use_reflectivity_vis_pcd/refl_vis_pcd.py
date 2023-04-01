import open3d as o3d
import numpy as np
# debug，打印工作目录
import sys,os
print(os.getcwd())
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 读取点云数据和反射率值
pcd = o3d.io.read_point_cloud("./test_net_model_py/point_cloud_npy_2_pcd/coor_1.pcd")
refl = np.asarray(pcd.colors)[:, 0] # 反射率值存储在colors属性的红色通道中

# 将反射率值映射到颜色空间中
color_map = o3d.visualization.ColorMapJet()
colors = color_map(np.clip(refl, 0.0, 1.0))[:,:-1]

# 设置点云的颜色
pcd.colors = o3d.utility.Vector3dVector(colors)

# 可视化点云
o3d.visualization.draw_geometries([pcd])