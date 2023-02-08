from utils import registry

# 创建一个名叫"dataset"的注册器常量对象，可以用于向这个注册器的字典中添加一些函数、类（模型）
"""
注册器中添加模块的示例：
 Example:
            >>> backbones = Registry('backbone')
            >>> @backbones.register_module()
            >>> class ResNet:
            >>>     pass
这里
 >>> @backbones.register_module() # 也可以直接在里面指定所注册的模块module的name
 >>> class ResNet:
 >>>     pass
 这三行代码相当于执行：
 >>> register_module('ResNet')
 注册器的字典中将会添加元素：{'ResNet': ResNet}
"""
DATASETS = registry.Registry('dataset')


def build_dataset_from_cfg(cfg, default_args = None):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT): 
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    # 由于dataset模块就存在于名叫dataset注册器对象DATASETS中，build函数的目的是从配置文件对应的config dict构建一个实例模块dataset
    return DATASETS.build(cfg, default_args = default_args)


