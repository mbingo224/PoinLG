from utils import registry


MODELS = registry.Registry('models') # 从注册器构建模型实例，初始model的name是：models


def build_model_from_cfg(cfg, **kwargs):
    """
    Build a dataset, defined by `dataset_name`.
    Args:
        cfg (eDICT): 
    Returns:
        Dataset: a constructed dataset specified by dataset_name.
    """
    return MODELS.build(cfg, **kwargs)


