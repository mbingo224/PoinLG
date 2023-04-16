CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/dist_train.sh 2 13232 \
    --config ./cfgs/ShapeNet55_models/PoinTr.yaml \
    --exp_name Experiments_8_bs_48