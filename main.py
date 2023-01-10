from tools import run_net
from tools import test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter

def main():
    # args 获取命令行的参数并解析，实质上大部分参数都是读取./cfgs/目录下的配置文件，不同的数据集对应不同的配置文件，例如对于PCN：./cfgs/PCN_models/PoinTr.yaml
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        # 设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    
    if args.launcher == 'none': # 单机多卡分布式训练DP
        args.distributed = False
    else: # 当args.launcher == 'pytorch' 采用多机多卡分布式训练DDP，使用前需要完成多进程的初始化，一般采用这种方式，训练速度会快一些
        args.distributed = True
        # DDP分布式训练环境初始化
        # world_size, rank, local_rank, rank。world size指进程总数，在这里就是我们使用的卡数；
        # rank指进程序号，local_rank指本地序号，两者的区别在于前者用于进程间通讯，后者用于本地设备分配
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # 对于PCN作为训练集，args.experiment_path = "./experiments/PoinTr/PCN_models/example"
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log') # 日志目录
    logger = get_root_logger(log_file=log_file, name=args.log_name) # 配置根日志器及其子日志器的处理器、格式器，对于PCN数据集对应的配置文件中args.log_name一般是PoinTr
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train')) # 指定训练时模型的输出事件的写入路径
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger = logger)
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank() 

    # run
    if args.test:
        test_net(args, config)
    else:
        run_net(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
