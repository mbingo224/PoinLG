from .runner import run_net # 对于tools包 __init__.py会先将runner模块下的run_net函数导入，包外面再想导入这些内容时，就可以用from tools import 函数名来导入，main.py中导入即是这样使用的
from .runner import test_net