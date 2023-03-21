import numpy as np
import os
import torch


coor = torch.randn(1, 3, 128)
f = torch.randn(1, 128, 128)
data_list = [coor, f]
current_path = os.getcwd()
file_path = os.path.join(current_path, 'test_net_model_py/point_cloud_npy_2_pcd/')
np.save(os.path.join(file_path, 'coor.npy'), data_list[0].numpy())
np.save(os.path.join(file_path, 'f.npy'), data_list[1].numpy())
