import logging
import torch.distributed as dist

logger_initialized = {}

def get_root_logger(log_file=None, log_level=logging.INFO, name='main'):
    """Get root logger and add a keyword filter to it.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet3d".
    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword. Defaults to 'mmdet3d'.
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    # logging默认就有一个root的Logger对象，通过logging.root可以查询到默认为level为warning级别
    # 单独创建Logger对象（默认是root logger的子对象），在logger进行初始化的时候如果未指定文件名filename即log_file参数未指定，将会使用StreamHandler指定初始化的文件流
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)
    # add a logging filter 
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    # 创建Logger日志输出的对象，getLogger返回具有指定 name 的日志记录器，getLogger() 返回对具有指定名称的记录器实例的引用（如果已提供），或者如果没有则返回 root 
    # 所有用给定的 name 对该函数的调用都将返回相同的日志记录器实例(重复调用这个函数将会获得相同的logger)。 这意味着日志记录器实例不需要在应用的各部分间传递。
    logger = logging.getLogger(name)
    # 这里如果logger日志器已经初始化了，就直接返回ilogger
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    # 父logger初始化了以后，子logger将不会再重复初始化
    for logger_name in logger_initialized:
        # startswith() 方法用于检查字符串name是否是以指定子字符串logger_name开头，如果是则返回 True，否则返回 False
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    # 处理控制台的重复日志 从 1.8.0 开始，PyTorch DDP 将 StreamHandler <stderr> (NOTSET) 附加到根记录器
    # 判断root Logger 的Handler，并进行日志级别设置，将错误或更高的所有日志消息发送到标准输出
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    # 创建StreamHandler实例，无参——由于未指定stream，则默认将日志记录输出sys.stderr标准错误输出流，也就是输出到控制台
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    # 对于分布式训练DDP进行设置，返回当前进程组的rank（可理解为进行ID一样地标识符）
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # DP分布式训练场景下
    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        # 创建一个FileHandler对象（继承自StreamHandler），将打开指定的文件并将其用作日志记录流
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler) # 将FileHandler对象加入handlers处理器列表中

    # 创建格式器对象formatter，以可读的格式记录时间（本地时间）、logger的name(在这里一般被设置为PoinTr)、消息的严重性以及消息的内容顺序进行格式输出
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 分别对子logger对象所添加的不同处理器Handler设置格式器formatter，这里当rank = 0 时logger对象相当于拥有2个处理器Handler(StreamHandler 和 FileHandler)
    # 如果是两个handler处理器，那么希望处理器FileHandler将所有级别为INFO及以上的日志消息发送到日志文件，同时StreamHandler处理器也将所有级别为INFO及以上的日志消息发送到标准输出
    for handler in handlers: 
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0: # 设置日志器logger的level为INFO的level
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)
    # logger已初始化完成，需置位，防止重复初始化
    logger_initialized[name] = True


    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.
    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')