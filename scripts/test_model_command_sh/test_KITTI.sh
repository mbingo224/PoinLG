bash ./scripts/test.sh 0 \
    --ckpts ./experiments/PoinTr/KITTI_models/Finetune_Experiments_8_bs_38/ckpt-best.pth \
    --config ./cfgs/KITTI_models/PoinTr.yaml \
    --exp_name Finetune_Experiments_8_bs_38

# KITTI 评估时，需要将测试集的生成的npy点云数据转换为图片，然后再执行下面的程序去进行评估
CUDA_VISIBLE_DEVICES=0 python KITTI_metric.py \
    --vis ./experiments/PoinTr/KITTI_models/test_Finetune_Experiments_8_bs_38/vis_result # 这里的路径是上一步实验生成的测试目录下的可视化输出路径vis_result