import yaml
from easydict import EasyDict
import os
from .logger import print_log

def log_args_to_file(args, pre='args', logger=None):
    for key, val in args.__dict__.items():
        print_log(f'{pre}.{key} : {val}', logger = logger)

def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f'{pre}.{key} = edict()', logger = logger)
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        print_log(f'{pre}.{key} : {val}', logger = logger)

def merge_new_config(config, new_config): # 递归地将new_config的对象复制到config对象中
    for key, val in new_config.items(): # 迭代地处理new_config字典对象中的元素
        if not isinstance(val, dict): # 对非dict类型的value进行合并
            if key == '_base_':
                with open(new_config['_base_'], 'r') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict() # 为config字典对象添加键值对，并且value初始化为字典对象
                merge_new_config(config[key], val) # 递归地将new_config的对象复制到config对象中
            else:
                config[key] = val
                continue
        if key not in config: # 得先判断将要添加的key是否存在config对象中，不存在就先为config添加一个键值对
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config

def cfg_from_yaml_file(cfg_file):
    config = EasyDict() # EasyDict类可以方便地创建字典dict对象并且使得此对象访问其中的元素像实例对象访问属性那么简单，即使用“.”访问符
    with open(cfg_file, 'r') as f: # open函数运行后将会返回一个文件流对象 其将被赋值给变量f，无论期间是否抛出异常，都能保证 with as 语句执行完毕后自动关闭已经打开的文件
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader) # load函数将yaml配置文件中的数据转化为字典对象赋给new_config，加载时指定加载器FullLoader：加载完整的YAML语言
        except: # 如果出现异常，一般有可能是版本较旧导致的，那就切换到旧版的load加载函数
            new_config = yaml.load(f)
    merge_new_config(config=config, new_config=new_config)        
    return config

def get_config(args, logger=None): # 从命令行运行训练或测试网络中读取配置文件yaml
    if args.resume: # 自动恢复训练的标志,若训练被意外终止，可进行恢复
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path): # 判断配置文件是否存在，若不存在就报错，退出程序
            print_log("Failed to resume", logger = logger)
            raise FileNotFoundError()
        print_log(f'Resume yaml from {cfg_path}', logger = logger) # 将恢复训练时配置文件的获取的日志打印到日志器，”logger = logger“ 这种传参方式可以改变顺序，还可以起到默认值的作用
        args.config = cfg_path # 将从命令行运行训练或测试网络中获取的配置文件的路径参数"--config"更新为新生成的实验路径experiment_path来作为恢复训练的配置文件，例如args.config配置文件参数更改为："./experiments/PoinTr/PCN_models/example/config.yaml"
    config = cfg_from_yaml_file(args.config)
    if not args.resume and args.local_rank == 0: # 训练未被中断时，将命令行的config参数表示的路径配置文件复制到新生成的实验路径experiment_path
        save_experiment_config(args, config, logger)
    return config

def save_experiment_config(args, config, logger = None): # 将命令行的config参数表示的路径配置文件复制到新生成的实验路径experiment_path
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    # system函数可以将字符串转化成命令在服务器上运行；其原理是每一条system函数执行时，其会创建一个子进程在系统上执行命令行，子进程的执行结果无法影响主进程；
    os.system('cp %s %s' % (args.config, config_path)) # 将命令行的config参数表示的路径配置文件复制到新生成的实验路径experiment_path
    print_log(f'Copy the Config file from {args.config} to {config_path}',logger = logger )