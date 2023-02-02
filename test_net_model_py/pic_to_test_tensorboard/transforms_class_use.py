from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "../virtual_env/PoinTr/lib/python3.8/site-packages/sklearn/datasets/images/china.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs") # 设置模型输出事件events的生成目录，这里会在项目目录下创建这个logs目录

tensor_trans = transforms.ToTensor() # ToTensor这个class定义了__call__函数因此可以直接调用
tensor_img = tensor_trans(img)

writer.add_image("tensor_img", tensor_img)

writer.close()