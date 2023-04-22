CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/ShapeNet34_models/PoinTr.yaml \
    --exp_name Experiments_5_bs_64