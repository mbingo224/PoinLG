import torch
import numpy as np
import os 

pred_data = torch.from_numpy(np.load(os.path.join("/home/zjin/Desktop/PoinTr/experiments/PoinTr/KITTI_models/test_example/vis_result", "frame_0_car_0_000", 'input.npy'))).unsqueeze(0).cuda()
#print(type(pred_data))
print(pred_data.shape)
print(pred_data)