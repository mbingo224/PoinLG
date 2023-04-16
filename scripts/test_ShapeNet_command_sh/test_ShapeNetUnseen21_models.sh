bash ./scripts/test.sh 0 \
    --ckpts ./experiments/PoinTr/ShapeNet34_models/Experiments_8_bs_48/ckpt-best.pth \
    --config ./cfgs/ShapeNetUnseen21_models/PoinTr.yaml \
    --mode median \
    --exp_name Unseen21_Experiments_8_bs_48