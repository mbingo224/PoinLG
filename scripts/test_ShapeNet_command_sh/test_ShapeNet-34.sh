bash ./scripts/test.sh 0 \
    --ckpts ./experiments/PoinTr/ShapeNet34_models/Experiments_8_bs_48/ckpt-best.pth \
    --config ./cfgs/ShapeNet34_models/PoinTr.yaml \
    --mode hard \ # 调整测试模式easy，median，hard
    --exp_name Experiments_8_bs_48