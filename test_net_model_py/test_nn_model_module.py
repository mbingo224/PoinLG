import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 这里两个参数表示输入样本是一个8维的数据，输出数据是一个1维数据，结合上图这里是一个mini_batch那就是n个样本
        self.linear = nn.Linear(8, 1) 
        self.sigmoid = nn.Sigmoid() # 创建一个sigmoid函数对象
    def forward(self, x): # 前向传播
        x = self.sigmoid(self.linear(x)) # 经过线性模型linear创建实例对象
        return x
        
model = Model()
model(5)