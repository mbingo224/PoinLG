import os
import argparse
from pathlib import Path

# 解析命令行参数，并根据命令行参数进行异常处理，并创建放置训练生成的模型文件的目录，设置日志目录，将参数返回
def get_args():
    parser = argparse.ArgumentParser()
    '''
    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')
    '''
    # debug train or debug test
    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file',
        #default='./cfgs/PCN_models/PoinTr.yaml'
        default='./cfgs/ShapeNet55_models/PoinTr.yaml') 
    
    # choices=['none', 'pytorch']表示参数可以接受的值的范围，pytorch表示DDP的训练方式
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')     
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)   
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.') # action='store_true'当这一选项存在时，为 args.deterministic 赋值为 True，没有指定时则默认赋值为 False。      
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    
    # some args
    # debug train or debug test，修改实验生成文件夹默认为'example'
    # parser.add_argument('--exp_name', type = str, default='example', help = 'experiment name') # debug test
    # ShapeNet55
    parser.add_argument('--exp_name', type = str, default='Experiments_8_bs_38', help = 'experiment name') # debug test
    
    # 当想要仅微调某个已训练的模型，就可使用--start_ckpts参数来指定该模型的权重文件（包含了 model_state_dict 和 optimizer_state_dict ）以在其上继续训练
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path') 
    
    # debug test
    #parser.add_argument('--ckpts', type = str, default='./pretrained/concat_graph_feature_fail_best.pth', help = 'test used ckpt path') # debug test
    # ShapeNet55
    parser.add_argument('--ckpts', type = str, default='./experiments/PoinTr/ShapeNet55_models/Experiments_8_bs_48/ckpt-best.pth', help = 'test used ckpt path') # debug test
    
    # debug train，训练时不需要预加载预训练模型
    #parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
        
    # debug test，这里训练的时候理应把default值修改回 False，
    # 实际上测试或者训练这个参数的设置都是在script的脚本下去进行设置的，可参考：PoinTr/scripts/test.sh 
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=True, 
        help = 'test mode for certain ckpt') 
    
    # debug train
    '''parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    '''
    # debug test
    # ShapeNet55 easy模式
    parser.add_argument(
        '--mode', 
        choices=['easy', 'median', 'hard', None],
        #default=None,
        default='easy',
        help = 'difficulty mode for shapenet')        
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')
    # test时应加载一个ckpts权重文件来进行评估
    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    # 在操作系统中定义的环境变量，全部保存在os.environ这个变量中(字典的形式)，如果LOCAL_RANK并不在环境变量中，向其中添加LOCAL_RANK（进程在本地服务器节点上的序号表示）变量
    # os.environ获得每个进程的节点ip信息，全局rank以及local rank，有了这些就可以很方便很方便的完成DDP分布式训练的初始化
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' +args.mode

    # 额外的给参数解析对象添加三个属性参数experiment_path、tfboard_path和log_name
    # Path(args.config).stem将会返回当前路径中的主文件名，这里的当前路径即指args.config参数所表示的路径，例如对于args.config = "./cfgs/PCN_models/PoinTr.yaml"，
    # Path(args.config).stem将会取得 PoinTr，Path(args.config).parent.stem将会取得PCN_models，因此再拼接上args.exp_name = "example" 可得到 args.experiment_path = "./experiments/PoinTr/PCN_models/example"
    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args

def create_experiment_dir(args): # 创建模型实验experiment和tfboard目录
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

