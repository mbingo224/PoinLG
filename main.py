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
    # logger，根据当地时间戳来命名日志文件
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    # 对于PCN作为训练集，args.experiment_path = "./experiments/PoinTr/PCN_models/example"
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log') # 日志目录
    logger = get_root_logger(log_file=log_file, name=args.log_name) # 配置根日志器及其子日志器的处理器、格式器，对于PCN数据集对应的配置文件中args.log_name一般是PoinTr
    # define the tensorboard writer
    if not args.test: # 非测试test模式下
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train')) # 指定训练时模型的输出事件events的写入路径
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    
    # config # 从命令行运行训练或测试网络中读取配置文件yaml以获得各种配置参数
    config = get_config(args, logger = logger)
    # batch size，为config对象额外再添加训练train时的batch size，但是需要分成DDP和DP两种情况
    if args.distributed: # 分布式训练是true
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size # config.total_bs可能是world_size的倍数，因为在DDP训练时world_size指的是显卡的数量，因此分配给每张卡的batch size需要一致
    else:
        config.dataset.train.others.bs = config.total_bs
    
    # log 分别将参数和配置写入到文件的日志信息打印出来，意即是xx.log日志文件中的开始的一系列args和config信息
    # 因此即使是test模式下，config.dataset.train和config.dataset.val的日志也会被输出到日志文件中
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        # seed用于指定随机数生成时所用算法开始的整数值，这里还加上了local_rank作为初始seed，使用相同的seed值，则每次生成的随即数都相同，固定随机源即可保证模型的训练结果始终一致
        # deterministic设置所有算法需要保证其有确定性的实现，这样可保证每次网络的训练结果都相同
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank() # 断言当前进程所获得的全局rank与local_rank是否一致，如果不一致就抛出错误

    # run
    if args.test:
        test_net(args, config)
    else:
        run_net(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
