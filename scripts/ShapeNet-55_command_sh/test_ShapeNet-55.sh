bash ./scripts/test.sh 1 \
    --ckpts ./experiments/PoinTr/ShapeNet55_models/Retrain_Experiments_5_bs_68/ckpt-best.pth \
    --config ./cfgs/ShapeNet55_models/PoinTr.yaml \
    --mode median \
    --exp_name Retrain_Experiments_5_bs_68