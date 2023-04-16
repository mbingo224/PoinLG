bash ./scripts/test.sh 0 \
    --ckpts ./experiments/PoinTr/KITTI_models/Experiments_8_bs_38/ckpt-best.pth \
    --config ./cfgs/KITTI_models/PoinTr.yaml \
    --exp_name Experiments_8_bs_38
CUDA_VISIBLE_DEVICES=0 python KITTI_metric.py \
    --vis ./experiments/PoinTr/KITTI_models/test_Experiments_8_bs_38/vis_result # 这里的路径是上一步实验生成的测试目录下的可视化输出路径vis_result