CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/KITTI_models/PoinTr.yaml \
    --exp_name Finetune_Experiments_8_bs_38 \
    --start_ckpts ./experiments/PoinTr/PCN_models/Experiments_8__bs_36/ckpt-best.pth