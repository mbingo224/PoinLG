'''
import numpy as np
import matplotlib.pyplot as plt
depthmap = np.load('/home/zjin/Desktop/PoinTr/experiments/PoinTr/KITTI_models/test_example/vis_result/frame_0_car_0_000/pred.npy')    #使用numpy载入npy文件
plt.imshow(depthmap)              #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
# plt.colorbar()                   #添加colorbar
plt.savefig('depthmap.jpg')       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
plt.show()                        #在线显示图像
'''
import os
import numpy as np
np.set_printoptions(suppress=True) 
# 作用是取消numpy默认的科学计数法，测试表明open3d点云读取函数没法读取科学计数法的表示
import open3d as o3d
data = np.load('/home/zjin/Desktop/PoinTr/experiments/PoinTr/KITTI_models/test_example/vis_result/frame_0_car_0_000/pred.npy')
txt_data = np.savetxt('./vis_result/frame_0_car_0_000/pred_1.txt', data)
pcd = o3d.io.read_point_cloud('./vis_result/frame_0_car_0_000/pred.txt', format='xyz') # 一定要注意这里format格式一定要与.npy中的tesor的格式相匹配
print('pcd的类型:', type([pcd]))
# 此处因为npy里面正好是 x y z 的数据排列形式，所以format='xyz'
print('pcd的点云的数量:', pcd)
# 可视化窗口大小是：1200 X 600
o3d.visualization.draw_geometries([pcd], width=1200, height=600) # 可视化点云，这里pcd是添加了[]，是因为绘制的几何对象不止一个因此需要以列表list形式，
o3d.io.write_point_cloud('./vis_result/frame_0_car_0_000/pred.pcd', pcd) # 存储为pcd格式
# import numpy as np
# from open3d import*

# source_data = np.load('curtain_0088.npy')[:,0:3]  #以(n,3)的形状读取点云
# point_cloud = open3d.geometry.PointCloud() # <class 'open3d.geometry.PointCloud'>是open3d中点云的标准类型
# point_cloud.points = open3d.utility.Vector3dVector(source_data) # 将形状 （n， 3） 的 float64 numpy 数组转换为 Open3D 格式的class，open3d.geometry.PointCloud()构造的class的属性points表示：open3d中一个形状为 (num_points, 3) 的 float64 数组，使用 numpy.asarray() 来访问数据
# open3d.visualization.draw_geometries([point_cloud]) # 
