bash ./scripts/test.sh 0 \
    --ckpts ./experiments/PoinTr/ShapeNet55_models/Experiments_8_bs_48/ckpt-best.pth \
    --config ./cfgs/ShapeNet55_models/PoinTr.yaml \
    --mode easy \ # 有 easy, median, hard 三种模式，分别对应不同的测试集，最后求平均来作为最佳结果
    --exp_name Experiments_8_bs_48

bash ./scripts/test.sh 1 \
    --ckpts ./pretrained/PCNnew.pth \
    --config ./cfgs/PCN_models/PCN.yaml \
    --exp_name PCNnew

bash ./scripts/test.sh 1 \
    --ckpts ./experiments/PoinTr/ShapeNet55_models/Get_multi_model_Experiments_8/ckpt-epoch-066-SnowflakeNet.pth \
    --config ./cfgs/ShapeNet55_models/PoinTr.yaml \
    --mode easy \
    --exp_name SnowflakeNet_Visual
