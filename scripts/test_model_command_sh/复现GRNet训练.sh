CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/PCN_models/PoinTr.yaml \
    --exp_name 复现GRNet训练 \
    --start_ckpts ./experiments/PoinTr/PCN_models/Experiments_11_bs_44_num_query=256/ckpt-last.pth